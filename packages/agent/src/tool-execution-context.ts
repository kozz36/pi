/** Host policy for one invocation. Never populate this from model arguments. */
export type ToolExecutionDecision = "allow" | "deny" | "confirm";

export interface ToolExecutionAdmission {
	isValid: () => boolean;
	/** Evaluate every final plan. Omission means confirmation is required. */
	evaluate?: (plan: unknown, presentationContext?: unknown) => ToolExecutionDecision | Promise<ToolExecutionDecision>;
	approve: (plan: unknown, presentationContext?: unknown) => Promise<boolean>;
}

/** Per-invocation state passed only to opted-in executors. */
export interface ToolExecutionContext {
	assertValid(executor?: object): void;
	admit(plan: unknown, presentationContext?: unknown, canConfirm?: () => boolean): Promise<void>;
}

// Capability bindings, not approvals. Bindings identify actual function objects.
type NativeInvocationScope = <T>(context: ToolExecutionContext | undefined, invoke: () => Promise<T>) => Promise<T>;

interface Binding {
	valid: () => boolean;
	delegate?: object;
	scope?: NativeInvocationScope;
}
const bindings = new WeakMap<object, Binding>();
// Clear every known carrier on entry, including unbound legacy calls nested in
// guarded calls. Carriers hold lifecycle state, never ambient approval.
const nativeScopes = new Set<NativeInvocationScope>();

/**
 * Privileged native backend registration. Not a sandbox against extension code.
 * The backend must await admit at its final effect boundary and execute once.
 * An optional native scope must isolate async work, clear on undefined, and restore
 * the caller on return/throw. It is lifecycle evidence, never ambient approval.
 */
export function registerToolExecutionContext(
	executor: object,
	isValid: () => boolean,
	scope?: NativeInvocationScope,
): void {
	if (typeof executor !== "function") throw new Error("Expected an executor function");
	bindings.set(executor, { valid: isValid, scope });
	if (scope) nativeScopes.add(scope);
}

/** Bind a known forwarding function to its exact delegate, not metadata or a tool ID. */
export function bindToolExecutionContext(executor: object, delegate: object, isValid: () => boolean): void {
	const binding = bindings.get(delegate);
	if (!binding) return;
	if (typeof executor !== "function") throw new Error("Expected an executor function");
	bindings.set(executor, {
		delegate,
		scope: binding.scope,
		valid: () => bindings.get(delegate) === binding && binding.valid() && isValid(),
	});
}

/** @internal Enter before forwarding; the native carrier owns async isolation/restoration. */
export function runInToolExecutionScope<T>(
	executor: object,
	context: ToolExecutionContext | undefined,
	invoke: () => Promise<T>,
): Promise<T> {
	context?.assertValid(executor);
	const selected = bindings.get(executor)?.scope;
	const scopes = [...nativeScopes];
	const enter = (index: number): Promise<T> => {
		const scope = scopes[index];
		return scope ? scope(scope === selected ? context : undefined, () => enter(index + 1)) : invoke();
	};
	return enter(0);
}

class InvocationContext implements ToolExecutionContext {
	#consumed = false;
	#pending = false;
	#active = true;
	#valid: () => boolean;
	#executors: readonly object[];
	#admission: ToolExecutionAdmission;

	constructor(admission: ToolExecutionAdmission, valid: () => boolean, executors: readonly object[]) {
		// Pin callback identities and their receiver while retaining live policy state.
		this.#admission = {
			isValid: admission.isValid.bind(admission),
			evaluate: admission.evaluate?.bind(admission),
			approve: admission.approve.bind(admission),
		};
		this.#valid = valid;
		this.#executors = executors;
	}

	assertValid(executor?: object): void {
		if (executor && !this.#executors.includes(executor))
			throw new Error("Executor is outside the bound admission path");
		if (!this.#active || !this.#valid() || !this.#admission.isValid()) {
			throw new Error("Tool execution admission is stale or cancelled");
		}
	}

	async admit(plan: unknown, presentationContext?: unknown, canConfirm: () => boolean = () => true): Promise<void> {
		this.assertValid();
		if (this.#pending || this.#consumed) throw new Error("Tool execution admission already used");
		this.#pending = true;
		const decision = await (this.#admission.evaluate?.(plan, presentationContext) ?? "confirm");
		this.assertValid();
		if (decision !== "allow") {
			if (decision !== "confirm") throw new Error("Final execution policy denied");
			if (!canConfirm()) throw new Error("Final admission requires UI context");
			if ((await this.#admission.approve(plan, presentationContext)) !== true) {
				throw new Error("Tool execution admission denied");
			}
			this.assertValid();
			if (!canConfirm()) throw new Error("Final admission UI is no longer available");
		}
		this.#consumed = true;
	}

	finish(): void {
		this.assertValid();
		if (!this.#consumed) throw new Error("Executor did not request final admission");
	}

	close(): void {
		this.#active = false;
	}
}

/** @internal Dispatcher-owned construction. */
export function createToolExecutionContext(
	executor: object,
	admission: ToolExecutionAdmission,
	valid: () => boolean,
): InvocationContext {
	const binding = bindings.get(executor);
	if (!binding) throw new Error("Executor does not support final admission");
	const executors: object[] = [];
	let current: object | undefined = executor;
	while (current) {
		if (executors.includes(current)) throw new Error("Cyclic execution context binding");
		executors.push(current);
		current = bindings.get(current)?.delegate;
	}
	const context = new InvocationContext(
		admission,
		() => bindings.get(executor) === binding && binding.valid() && valid(),
		executors,
	);
	context.assertValid();
	return context;
}

export function isToolExecutionContext(value: unknown): value is ToolExecutionContext {
	return value instanceof InvocationContext;
}
