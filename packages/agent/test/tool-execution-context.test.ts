import { AsyncLocalStorage } from "node:async_hooks";
import { createAssistantMessageEventStream, type AssistantMessage } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { expect, it, vi } from "vitest";
import {
	Agent,
	runToolCall,
	runAgentLoop,
	bindToolExecutionContext,
	type AgentEvent,
	type AgentOptions,
	type AgentLoopConfig,
	type BeforeToolCallContext,
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

const lifecycleResult = { content: [{ type: "text" as const, text: "done" }], details: undefined };
function lifecycleMessage(names: string[]): AssistantMessage {
	return {
		role: "assistant", content: names.map((name) => ({ type: "toolCall", id: "reused", name, arguments: {} })),
		api: "openai-responses", provider: "openai", model: "fixture", stopReason: "toolUse", timestamp: 0,
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
	};
}
function lifecycleAgent(tools: AgentTool[], options: Partial<AgentOptions> = {}, names = tools.map((tool) => tool.name)) {
	const events: AgentEvent[] = [];
	const agent = new Agent({
		initialState: { tools },
		streamFn: () => {
			const stream = createAssistantMessageEventStream();
			queueMicrotask(() => stream.push({ type: "done", reason: "toolUse", message: lifecycleMessage(names) }));
			return stream;
		},
		finishTurn: () => ({ action: "end" }),
		...options,
	});
	agent.subscribe((event) => { events.push(event); });
	return { agent, events };
}
function lifecycleTool(
	execute: (this: AgentTool, ...args: Parameters<AgentTool["execute"]>) => ReturnType<AgentTool["execute"]>,
	name = "native",
): AgentTool {
	return { name, label: name, description: "fixture", parameters: Type.Object({}), execute };
}
function executionEnds(events: AgentEvent[]) {
	return events.filter((event) => event.type === "tool_execution_end");
}

it("public Agent rejects unsupported admission before executor or policy effects", async () => {
	const execute = vi.fn(async () => lifecycleResult);
	const evaluate = vi.fn(() => "allow" as const);
	const approve = vi.fn(async () => true);
	const { agent, events } = lifecycleAgent([lifecycleTool(execute)], {
		beforeToolCall: async () => ({ admission: { isValid: () => true, evaluate, approve } }),
	});
	await agent.prompt("fixture");
	expect(execute).not.toHaveBeenCalled();
	expect(evaluate).not.toHaveBeenCalled();
	expect(approve).not.toHaveBeenCalled();
	expect(executionEnds(events)[0]).toMatchObject({ isError: true,
		result: { content: [{ type: "text", text: "Executor does not support final admission" }] } });
});

it.each(["allow", "deny", "confirm"] as const)("public Agent awaits %s on the actual fifth context and closes it", async (decision) => {
	let effects = 0;
	let captured: ToolExecutionContext | undefined;
	const approve = vi.fn(async () => true);
	const tool = lifecycleTool(async function (_id, _args, _signal, update, context) {
		expect(this).toBe(tool);
		expect(arguments.length).toBe(5);
		expect(isToolExecutionContext(context)).toBe(true);
		captured = context;
		await context!.admit({ final: true });
		effects++;
		update?.(lifecycleResult);
		return lifecycleResult;
	});
	registerToolExecutionContext(tool.execute, () => true);
	const { agent, events } = lifecycleAgent([tool], {
		beforeToolCall: async () => ({ admission: { isValid: () => true, evaluate: () => decision, approve } }),
	});
	await agent.prompt("fixture");
	expect(effects).toBe(decision === "deny" ? 0 : 1);
	expect(approve).toHaveBeenCalledTimes(decision === "confirm" ? 1 : 0);
	expect(executionEnds(events)[0].isError).toBe(decision === "deny");
	expect(() => captured!.assertValid()).toThrow("stale");
	expect(events.filter((event) => event.type === "tool_execution_update")).toHaveLength(decision === "deny" ? 0 : 1);
});

it("public Agent gives hard blocks precedence over unsupported admission", async () => {
	const execute = vi.fn(async () => lifecycleResult);
	const evaluate = vi.fn(() => "allow" as const);
	const { agent, events } = lifecycleAgent([lifecycleTool(execute)], {
		beforeToolCall: async () => ({ block: true, reason: "host block", terminate: true,
			admission: { isValid: () => true, evaluate, approve: async () => true } }),
	});
	await agent.prompt("fixture");
	expect(execute).not.toHaveBeenCalled();
	expect(evaluate).not.toHaveBeenCalled();
	expect(executionEnds(events)[0]).toMatchObject({ isError: true, result: {
		content: [{ type: "text", text: "host block" }], terminate: true } });
});

it.each(["abort", "owner", "executor", "policy", "deny"])("public Agent prevents effects after %s during approval", async (change) => {
	let effects = 0;
	let valid = true;
	let agent: Agent;
	const policy = new ReceiverPolicy();
	policy.denied = false;
	policy.requiresConfirmation = true;
	policy.isValid = () => valid;
	policy.approve = async function () {
		expect(this).toBe(policy);
		await Promise.resolve();
		if (change === "abort") agent.abort();
		if (change === "owner") agent.state.tools = [{ ...tool }];
		if (change === "executor") tool.execute = async () => lifecycleResult;
		if (change === "policy") valid = false;
		return change !== "deny";
	};
	const tool = lifecycleTool(async (_id, _args, _signal, _update, context) => {
		await context!.admit({});
		effects++;
		return lifecycleResult;
	});
	registerToolExecutionContext(tool.execute, () => true);
	const fixture = lifecycleAgent([tool], { beforeToolCall: async () => ({ admission: policy }) });
	agent = fixture.agent;
	await agent.prompt("fixture");
	expect(effects).toBe(0);
	expect(executionEnds(fixture.events)[0]).toMatchObject({ isError: true, result: {
		content: [{ type: "text", text: change === "deny" ? "Tool execution admission denied" : "Tool execution admission is stale or cancelled" }] } });
});

it("public Agent closes missing admission and outstanding approval when execute returns", async () => {
	let release!: (approved: boolean) => void;
	let pending!: Promise<void>;
	let effects = 0;
	const tool = lifecycleTool(async (_id, _args, _signal, _update, context) => {
		pending = context!.admit({}).then(() => { effects++; });
		// Attach a rejection handler before the intentionally unawaited approval settles.
		void pending.catch(() => {});
		return lifecycleResult;
	});
	registerToolExecutionContext(tool.execute, () => true);
	const { agent, events } = lifecycleAgent([tool], { beforeToolCall: async () => ({ admission: {
		isValid: () => true, approve: () => new Promise((resolve) => { release = resolve; }),
	} }) });
	await agent.prompt("fixture");
	expect(executionEnds(events)[0]).toMatchObject({ isError: true, result: {
		content: [{ type: "text", text: "Executor did not request final admission" }] } });
	release(true);
	await expect(pending).rejects.toThrow("stale");
	expect(effects).toBe(0);
});

it("public Agent isolates parallel reused IDs, clears nested legacy scope and restores after errors", async () => {
	const carrier = new AsyncLocalStorage<ToolExecutionContext | undefined>();
	const contexts: ToolExecutionContext[] = [];
	const legacy = lifecycleTool(async function () {
		expect(this).toBe(legacy);
		expect(arguments.length).toBe(4);
		expect(carrier.getStore()).toBeUndefined();
		return lifecycleResult;
	}, "legacy");
	const tool = lifecycleTool(async (_id, _args, signal, _update, context) => {
		expect(carrier.getStore()).toBe(context);
		contexts.push(context!);
		await Promise.resolve();
		await runToolCall({ type: "toolCall", id: "reused", name: "legacy", arguments: {} }, {
			tools: [legacy], context: { messages: [], tools: [legacy] }, assistantMessage: lifecycleMessage([]), signal,
		});
		expect(carrier.getStore()).toBe(context);
		await context!.admit({});
		throw new Error("native fixture failure");
	});
	registerToolExecutionContext(tool.execute, () => true, (context, invoke) => carrier.run(context, invoke));
	const { agent, events } = lifecycleAgent([tool], { beforeToolCall: async () => ({ admission: {
		isValid: () => true, evaluate: () => "allow", approve: async () => true,
	} }) }, ["native", "native"]);
	await agent.prompt("fixture");
	await agent.prompt("fixture again");
	expect(new Set(contexts).size).toBe(4);
	expect(executionEnds(events)).toHaveLength(4);
	expect(executionEnds(events).every((event) => event.isError)).toBe(true);
	for (const context of contexts) expect(() => context.assertValid()).toThrow("stale");
	expect(carrier.getStore()).toBeUndefined();
});

it("public low-level runToolCall rejects a replaced current callable during approval", async () => {
	let effects = 0;
	const tool = lifecycleTool(async (_id, _args, _signal, _update, context) => {
		await context!.admit({});
		effects++;
		return lifecycleResult;
	});
	const tools = [tool];
	registerToolExecutionContext(tool.execute, () => true);
	const outcome = await runToolCall({ type: "toolCall", id: "reused", name: "native", arguments: {} }, {
		tools, context: { tools, messages: [] }, assistantMessage: lifecycleMessage([]),
		beforeToolCall: async () => ({ admission: { isValid: () => true, approve: async () => {
			await Promise.resolve();
			tools[0] = { ...tool };
			return true;
		} } }),
	});
	expect(effects).toBe(0);
	expect(outcome).toMatchObject({ isError: true, result: {
		content: [{ type: "text", text: "Tool execution admission is stale or cancelled" }] } });
});

it("public AgentLoop checks the caller's current owner, not only its copied context", async () => {
	let effects = 0;
	const tool = lifecycleTool(async (_id, _args, _signal, _update, context) => {
		await context!.admit({});
		effects++;
		return lifecycleResult;
	});
	registerToolExecutionContext(tool.execute, () => true);
	const context = { messages: [], tools: [tool] };
	const events: AgentEvent[] = [];
	await runAgentLoop([], context, {
		model: new Agent({ streamFn: () => { throw new Error("unused"); } }).state.model,
		convertToLlm: () => [], finishTurn: () => ({ action: "end" }),
		beforeToolCall: async () => ({ admission: { isValid: () => true, approve: async () => {
			await Promise.resolve();
			context.tools = [{ ...tool }];
			return true;
		} } }),
	}, (event) => { events.push(event); }, undefined, () => {
		const stream = createAssistantMessageEventStream();
		queueMicrotask(() => stream.push({ type: "done", reason: "toolUse", message: lifecycleMessage(["native"]) }));
		return stream;
	});
	expect(effects).toBe(0);
	expect(executionEnds(events)[0]).toMatchObject({ isError: true, result: {
		content: [{ type: "text", text: "Tool execution admission is stale or cancelled" }] } });
});

it("public Agent rejects stale policy before entering a supported executor", async () => {
	const execute = vi.fn(async () => lifecycleResult);
	registerToolExecutionContext(execute, () => true);
	const approve = vi.fn(async () => true);
	const { agent, events } = lifecycleAgent([lifecycleTool(execute)], {
		beforeToolCall: async () => ({ admission: { isValid: () => false, approve } }),
	});
	await agent.prompt("fixture");
	expect(execute).not.toHaveBeenCalled();
	expect(approve).not.toHaveBeenCalled();
	expect(executionEnds(events)[0].isError).toBe(true);
});

it("public Agent reports missing admission and closes an unused captured context", async () => {
	let captured: ToolExecutionContext | undefined;
	const tool = lifecycleTool(async (_id, _args, _signal, _update, context) => {
		captured = context;
		return lifecycleResult;
	});
	registerToolExecutionContext(tool.execute, () => true);
	const { agent, events } = lifecycleAgent([tool], { beforeToolCall: async () => ({ admission: {
		isValid: () => true, approve: async () => true,
	} }) });
	await agent.prompt("fixture");
	expect(executionEnds(events)[0]).toMatchObject({ isError: true, result: {
		content: [{ type: "text", text: "Executor did not request final admission" }] } });
	await expect(captured!.admit({})).rejects.toThrow("stale");
});

it("public Agent keeps legacy receiver, four args, hooks, results and late update filtering", async () => {
	let lateUpdate: (() => void) | undefined;
	const tool = lifecycleTool(async function (_id, _args, _signal, update) {
		expect(this).toBe(tool);
		expect(arguments.length).toBe(4);
		update?.(lifecycleResult);
		lateUpdate = () => update?.(lifecycleResult);
		return { ...lifecycleResult, structuredContent: { original: true }, terminate: true };
	});
	const afterToolCall = vi.fn(async () => ({ content: [{ type: "text" as const, text: "hooked" }] }));
	const { agent, events } = lifecycleAgent([tool], { afterToolCall });
	await agent.prompt("fixture");
	lateUpdate!();
	expect(afterToolCall).toHaveBeenCalledOnce();
	expect(events.filter((event) => event.type === "tool_execution_update")).toHaveLength(1);
	expect(executionEnds(events)[0]).toMatchObject({ isError: false,
		result: { content: [{ type: "text", text: "hooked" }], terminate: true } });
	expect(executionEnds(events)[0].result.structuredContent).toBeUndefined();
	expect(events.at(-1)?.type).toBe("agent_end");
});

it.each([false, true])("public Agent retains original hook identities and their actual shared receiver, dependent=%s", async (dependent) => {
	let effects = 0;
	let beforeReceiver: AgentLoopConfig | undefined;
	let afterReached = false;
	let beforeSignal: AbortSignal | undefined;
	const tool = lifecycleTool(async () => { effects++; return lifecycleResult; });
	async function beforeToolCall(this: AgentLoopConfig, _context: BeforeToolCallContext, signal?: AbortSignal) {
		beforeReceiver = this;
		beforeSignal = signal;
		expect(arguments.length).toBe(2);
		expect(signal).toBeInstanceOf(AbortSignal);
		if (dependent && this.model.id !== "unknown") return { block: true };
		return undefined;
	}
	async function afterToolCall(this: AgentLoopConfig, _context: unknown, signal?: AbortSignal) {
		afterReached = true;
		expect(arguments.length).toBe(2);
		expect(signal).toBe(beforeSignal);
		expect(this).toBe(beforeReceiver);
		expect(this.beforeToolCall).toBe(beforeToolCall);
		expect(this.afterToolCall).toBe(afterToolCall);
		return undefined;
	}
	const { agent, events } = lifecycleAgent([tool], { beforeToolCall, afterToolCall });
	await agent.prompt("fixture");
	expect(effects).toBe(1);
	expect(afterReached).toBe(true);
	expect(beforeReceiver).toBeDefined();
	expect(beforeReceiver!.beforeToolCall).toBe(beforeToolCall);
	expect(beforeReceiver!.afterToolCall).toBe(afterToolCall);
	expect(executionEnds(events)[0]).toMatchObject({ isError: false, result: lifecycleResult });
});

it.each(["hook", "session"])("public Agent invalidates %s replacement during approval without policy revocation", async (change) => {
	let effects = 0;
	let captured: ToolExecutionContext | undefined;
	let agent: Agent;
	const tool = lifecycleTool(async (_id, _args, _signal, _update, context) => {
		captured = context;
		await context!.admit({});
		effects++;
		return lifecycleResult;
	});
	registerToolExecutionContext(tool.execute, () => true);
	const fixture = lifecycleAgent([tool], { beforeToolCall: async () => ({ admission: {
		isValid: () => true,
		approve: async () => {
			await Promise.resolve();
			if (change === "hook") agent.beforeToolCall = async () => undefined;
			else agent.sessionId = "replacement";
			return true;
		},
	} }) });
	agent = fixture.agent;
	await agent.prompt("fixture");
	expect(effects).toBe(0);
	expect(() => captured!.assertValid()).toThrow("stale");
	expect(executionEnds(fixture.events)[0]).toMatchObject({ isError: true, result: {
		content: [{ type: "text", text: "Tool execution admission is stale or cancelled" }] } });
});
