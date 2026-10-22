import { AsyncLocalStorage } from "node:async_hooks";
import { Type } from "typebox";
import { expect, it, vi } from "vitest";
import {
	bindToolExecutionContext,
	isToolExecutionContext,
	registerToolExecutionContext,
	type AgentTool,
	type ToolExecutionAdmission,
	type ToolExecutionContext,
	type ToolExecutionDecision,
} from "../src/index.ts";
import { createToolExecutionContext, runInToolExecutionScope } from "../src/tool-execution-context.ts";

function invocation(admission: Partial<ToolExecutionAdmission> = {}, valid = () => true) {
	const executor = async () => {};
	registerToolExecutionContext(executor, () => true);
	return createToolExecutionContext(executor, {
		isValid: () => true,
		evaluate: () => "allow",
		approve: async () => true,
		...admission,
	}, valid);
}

class ReceiverPolicy implements ToolExecutionAdmission {
	denied = true;
	revoked = false;
	confirmed = false;
	requiresConfirmation = false;
	revokeOnApproval = false;
	evaluations = 0;
	approvals = 0;

	isValid(): boolean {
		return !this.revoked;
	}

	evaluate(): ToolExecutionDecision {
		this.evaluations++;
		return this.denied ? "deny" : this.requiresConfirmation ? "confirm" : "allow";
	}

	async approve(): Promise<boolean> {
		this.approvals++;
		await Promise.resolve();
		if (this.revokeOnApproval) this.revoked = true;
		return this.confirmed;
	}
}

it("preserves a prototype policy's denial without prompting or executing effects", async () => {
	const policy = new ReceiverPolicy();
	const executor = async () => {};
	registerToolExecutionContext(executor, () => true);
	const context = createToolExecutionContext(executor, policy, () => true);
	let effects = 0;
	expect(policy.evaluate()).toBe("deny");
	await expect(context.admit({}).then(() => { effects++; })).rejects.toThrow("policy denied");
	expect(effects).toBe(0);
	expect(policy.approvals).toBe(0);
	expect(policy.evaluations).toBe(2);
});

it("reads live revocation from the original prototype policy receiver", () => {
	const policy = new ReceiverPolicy();
	const executor = async () => {};
	registerToolExecutionContext(executor, () => true);
	const context = createToolExecutionContext(executor, policy, () => true);
	policy.revoked = true;
	expect(policy.isValid()).toBe(false);
	expect(() => context.assertValid()).toThrow("stale");
});

it.each([false, true])("pins every original policy callback while retaining live approval state, revoked=%s", async (revoked) => {
	const policy = new ReceiverPolicy();
	const executor = async () => {};
	registerToolExecutionContext(executor, () => true);
	const context = createToolExecutionContext(executor, policy, () => true);
	policy.isValid = () => { throw new Error("replaced validity callback"); };
	policy.evaluate = () => { throw new Error("replaced evaluation callback"); };
	policy.approve = async () => { throw new Error("replaced approval callback"); };
	policy.denied = false;
	policy.confirmed = true;
	policy.requiresConfirmation = true;
	policy.revokeOnApproval = revoked;
	let effects = 0;
	const pending = context.admit({}).then(() => { effects++; });
	if (revoked) await expect(pending).rejects.toThrow("stale");
	else await pending;
	expect(effects).toBe(revoked ? 0 : 1);
	expect(policy.evaluations).toBe(1);
	expect(policy.approvals).toBe(1);
	expect(policy.revoked).toBe(revoked);
});

it("requires registered callable owners, not metadata or copied function properties", () => {
	expect(() => registerToolExecutionContext({}, () => true)).toThrow("function");
	const native = async () => {};
	registerToolExecutionContext(native, () => true);
	const copy = Object.assign(async () => {}, native);
	expect(() => createToolExecutionContext(copy, { isValid: () => true, approve: async () => true }, () => true))
		.toThrow("does not support");
	bindToolExecutionContext(copy, async () => {}, () => true);
	expect(() => createToolExecutionContext(copy, { isValid: () => true, approve: async () => true }, () => true))
		.toThrow("does not support");
	expect(() => bindToolExecutionContext({}, native, () => true)).toThrow("function");
});

it("pins the delegate path and invalidates replaced registrations and live owners", () => {
	const native = async () => {};
	const wrapper = async () => {};
	let owner = true;
	registerToolExecutionContext(native, () => true);
	bindToolExecutionContext(wrapper, native, () => owner);
	const admission = { isValid: () => true, approve: async () => true };
	const context = createToolExecutionContext(wrapper, admission, () => true);
	expect(() => context.assertValid(native)).not.toThrow();
	expect(() => context.assertValid(async () => {})).toThrow("outside");
	owner = false;
	expect(() => context.assertValid()).toThrow("stale");
	owner = true;
	registerToolExecutionContext(native, () => true);
	expect(() => context.assertValid()).toThrow("stale");
});

it.each(["close", "abort", "policy"])("rejects %s during asynchronous evaluation", async (mode) => {
	const abort = new AbortController();
	let policyValid = true;
	let release!: (decision: "allow") => void;
	const context = invocation({
		isValid: () => policyValid,
		evaluate: () => new Promise((resolve) => { release = resolve; }),
	}, () => !abort.signal.aborted);
	const pending = context.admit({ command: "local fixture" });
	if (mode === "close") context.close();
	if (mode === "abort") abort.abort();
	if (mode === "policy") policyValid = false;
	release("allow");
	await expect(pending).rejects.toThrow("stale");
});

it("requires final admission before finish and allows exactly one attempt, including pending or denied attempts", async () => {
	let release!: (decision: "allow") => void;
	const context = invocation({ evaluate: () => new Promise((resolve) => { release = resolve; }) });
	expect(() => context.finish()).toThrow("did not request");
	const pending = context.admit({});
	await expect(context.admit({})).rejects.toThrow("already used");
	release("allow");
	await pending;
	expect(() => context.finish()).not.toThrow();
	await expect(context.admit({})).rejects.toThrow("already used");
	const denied = invocation({ evaluate: () => "deny" });
	await expect(denied.admit({})).rejects.toThrow("policy denied");
	await expect(denied.admit({})).rejects.toThrow("already used");
});

it("evaluates the final plan, pins policy callbacks, and does not prompt for allow or deny", async () => {
	const approve = vi.fn(async () => true);
	const evaluate = vi.fn(() => "allow" as const);
	const admission: ToolExecutionAdmission = { isValid: () => true, evaluate, approve };
	const native = async () => {};
	registerToolExecutionContext(native, () => true);
	const context = createToolExecutionContext(native, admission, () => true);
	admission.evaluate = () => "deny";
	const plan = Object.freeze({ command: "final", cwd: "/fixture" });
	const presentation = { label: "preview" };
	await context.admit(plan, presentation, () => false);
	expect(evaluate).toHaveBeenCalledWith(plan, presentation);
	expect(approve).not.toHaveBeenCalled();
	await expect(invocation({ evaluate: () => "deny", approve }).admit(plan)).rejects.toThrow("policy denied");
	expect(approve).not.toHaveBeenCalled();
});

it("defaults to confirmation and checks UI and validity again after approval", async () => {
	const approve = vi.fn(async () => true);
	await invocation({ evaluate: undefined, approve }).admit({});
	expect(approve).toHaveBeenCalledOnce();
	await expect(invocation({ evaluate: undefined, approve }).admit({}, undefined, () => false))
		.rejects.toThrow("requires UI");
	await expect(invocation({ evaluate: undefined, approve: async () => false }).admit({}))
		.rejects.toThrow("admission denied");
	let ui = true;
	await expect(invocation({ evaluate: undefined, approve: async () => { ui = false; return true; } })
		.admit({}, undefined, () => ui)).rejects.toThrow("no longer available");
	let valid = true;
	await expect(invocation({ evaluate: undefined, approve: async () => { valid = false; return true; } }, () => valid)
		.admit({})).rejects.toThrow("stale");
});

it("recognizes only SDK-created contexts and keeps independent invocations separate", async () => {
	const first = invocation();
	const second = invocation();
	expect(isToolExecutionContext(first)).toBe(true);
	expect(isToolExecutionContext({ ...first })).toBe(false);
	first.close();
	expect(() => first.assertValid()).toThrow("stale");
	await second.admit({});
	expect(() => second.finish()).not.toThrow();
});

it.each([false, true])("restores native scope after nested legacy entry, throwing=%s", async (throws) => {
	const carrier = new AsyncLocalStorage<ToolExecutionContext | undefined>();
	const native = async () => {};
	registerToolExecutionContext(native, () => true, (context, invoke) => carrier.run(context, invoke));
	const wrapper = async () => {};
	bindToolExecutionContext(wrapper, native, () => true);
	const context = createToolExecutionContext(wrapper, { isValid: () => true, approve: async () => true }, () => true);
	await runInToolExecutionScope(wrapper, context, async () => {
		expect(carrier.getStore()).toBe(context);
		const nested = runInToolExecutionScope(async () => {}, undefined, async () => {
			expect(carrier.getStore()).toBeUndefined();
			if (throws) throw new Error("nested failure");
		});
		if (throws) await expect(nested).rejects.toThrow("nested failure");
		else await nested;
		expect(carrier.getStore()).toBe(context);
	});
	expect(carrier.getStore()).toBeUndefined();
	context.close();
	expect(() => runInToolExecutionScope(wrapper, context, async () => {})).toThrow("stale");
});

it("retains legacy four-argument tools and their receiver without requiring context", async () => {
	const tool: AgentTool = {
		name: "legacy", label: "legacy", description: "fixture", parameters: Type.Object({}),
		async execute(id, _params, _signal, _update) {
			expect(this).toBe(tool);
			expect(arguments.length).toBe(4);
			return { content: [{ type: "text", text: id }], details: undefined };
		},
	};
	expect((await tool.execute("same", {}, undefined, undefined)).content).toEqual([{ type: "text", text: "same" }]);
});
