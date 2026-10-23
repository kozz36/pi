import type * as ChildProcess from "node:child_process";
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentEvent, type AgentTool, type ToolExecutionAdmission } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import type { TSchema } from "typebox";
import { beforeEach, expect, it, vi } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { createEventBus } from "../src/core/event-bus.ts";
import { createExtensionRuntime, loadExtensionFromFactory } from "../src/core/extensions/loader.ts";
import { ExtensionRunner } from "../src/core/extensions/runner.ts";
import type { ExtensionFactory, ExtensionToolContext, ToolCallEventResult } from "../src/core/extensions/types.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { createInMemoryModelRegistry } from "./model-runtime-test-utils.ts";
import { type BashOperations, createBashTool, createBashToolDefinition, createLocalBashOperations } from "../src/core/tools/bash.ts";
import { createPowerShellTool } from "../src/core/tools/powershell.ts";
import { wrapToolDefinition } from "../src/core/tools/tool-definition-wrapper.ts";
import * as outputFiles from "../src/utils/output-files.ts";
import * as shell from "../src/utils/shell.ts";

vi.mock("child_process", async (importOriginal) => {
	const actual = await importOriginal<typeof ChildProcess>();
	return { ...actual, spawn: vi.fn(actual.spawn) };
});

let cwd: string;
beforeEach(() => {
	vi.restoreAllMocks();
	vi.mocked(spawn).mockClear();
	cwd = mkdtempSync(join(tmpdir(), "native-bash-"));
	writeFileSync(join(cwd, "sentinel"), "unchanged");
	vi.spyOn(outputFiles, "createOutputFileStream");
});

function invoke<TParams extends TSchema, TDetails>(
	tool: AgentTool<TParams, TDetails>,
	admission?: ToolExecutionAdmission,
	args = { command: "printf changed > sentinel", timeout: 2 },
) {
	const events: AgentEvent[] = [];
	const agent = new Agent({
		initialState: { tools: [tool] },
		beforeToolCall: admission ? async () => ({ admission }) : undefined,
		streamFn: () => {
			const stream = createAssistantMessageEventStream();
			const message: AssistantMessage = {
				role: "assistant", content: [{ type: "toolCall", id: "one", name: tool.name, arguments: args }],
				api: "openai-responses", provider: "openai", model: "fixture", stopReason: "toolUse", timestamp: 0,
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			};
			queueMicrotask(() => stream.push({ type: "done", reason: "toolUse", message }));
			return stream;
		},
		finishTurn: () => ({ action: "end" }),
	});
	agent.subscribe((event) => { events.push(event); });
	return { agent, events, run: () => agent.prompt("owned fixture") };
}

function result(events: AgentEvent[]) {
	const end = events.find((event) => event.type === "tool_execution_end");
	if (!end || end.type !== "tool_execution_end") throw new Error("missing tool result");
	return end;
}

function noEffects(events: AgentEvent[], message: string) {
	expect(result(events)).toMatchObject({ isError: true, result: { content: [{ type: "text", text: message }] } });
	expect(spawn).not.toHaveBeenCalled();
	expect(outputFiles.createOutputFileStream).not.toHaveBeenCalled();
	expect(events.filter((event) => event.type === "tool_execution_update")).toHaveLength(0);
	expect(readFileSync(join(cwd, "sentinel"), "utf8")).toBe("unchanged");
	expect(readdirSync(cwd)).toEqual(["sentinel"]);
}

// This is only a mocked UI host. Native authority always comes from Agent's dispatcher.
function uiTool(operations?: BashOperations) {
	const host = { cwd, hasUI: true, ui: {} } as ExtensionToolContext;
	const factory = vi.fn(() => host);
	const tool = wrapToolDefinition(createBashToolDefinition(cwd, { operations, exposeSessionEnvironment: false }), factory);
	expect(factory).not.toHaveBeenCalled();
	return { host, tool, factory };
}

it("admits the actual default Bash plan once before its single real spawn", async () => {
	const approve = vi.fn(async () => false);
	const evaluate = vi.fn((plan: unknown) => {
		expect(spawn).not.toHaveBeenCalled();
		expect(plan).toMatchObject({ kind: "local-bash", command: "printf allowed", cwd, timeoutMs: 2000 });
		expect(Object.isFrozen(plan)).toBe(true);
		return "allow" as const;
	});
	const fixture = invoke(createBashTool(cwd), { isValid: () => true, evaluate, approve }, { command: "printf allowed", timeout: 2 });
	await fixture.run();
	expect(result(fixture.events)).toMatchObject({ isError: false, result: { structuredContent: { output: "allowed", exit_code: 0 } } });
	expect(evaluate).toHaveBeenCalledTimes(1);
	expect(approve).not.toHaveBeenCalled();
	expect(spawn).toHaveBeenCalledTimes(1);
});

it("awaits confirmation exactly once with the actual lazy UI host", async () => {
	const { host, tool, factory } = uiTool();
	const approve = vi.fn(async (_plan: unknown, presentation: unknown) => {
		expect(presentation).toBe(host);
		expect(spawn).not.toHaveBeenCalled();
		await Promise.resolve();
		return true;
	});
	const fixture = invoke(tool, { isValid: () => true, approve });
	await fixture.run();
	expect(result(fixture.events).isError).toBe(false);
	expect(approve).toHaveBeenCalledTimes(1);
	expect(factory).toHaveBeenCalledTimes(1);
	expect(spawn).toHaveBeenCalledTimes(1);
	expect(readFileSync(join(cwd, "sentinel"), "utf8")).toBe("changed");
});

it("denies without spawn, updates or stored-output effects", async () => {
	const approve = vi.fn(async () => true);
	const fixture = invoke(createBashTool(cwd), { isValid: () => true, evaluate: () => "deny", approve });
	await fixture.run();
	noEffects(fixture.events, "Final execution policy denied");
	expect(approve).not.toHaveBeenCalled();
});

it("declines confirmation without effects", async () => {
	const { tool } = uiTool();
	const approve = vi.fn(async () => false);
	const fixture = invoke(tool, { isValid: () => true, approve });
	await fixture.run();
	noEffects(fixture.events, "Tool execution admission denied");
	expect(approve).toHaveBeenCalledTimes(1);
});

it("fails closed when confirmation has no real UI context", async () => {
	const approve = vi.fn(async () => true);
	const fixture = invoke(createBashTool(cwd), { isValid: () => true, approve });
	await fixture.run();
	noEffects(fixture.events, "Final admission requires UI context");
	expect(approve).not.toHaveBeenCalled();
});

it.each(["revoke", "cancel", "replace executor", "lose UI"] as const)("rejects %s during awaited approval", async (change) => {
	const operations = createLocalBashOperations();
	const { host, tool } = uiTool(operations);
	let valid = true;
	const replacement = vi.fn(async () => ({ exitCode: 0 }));
	const approve = vi.fn(async () => {
		await Promise.resolve();
		if (change === "revoke") valid = false;
		if (change === "cancel") fixture.agent.abort();
		if (change === "replace executor") operations.exec = replacement;
		if (change === "lose UI") host.hasUI = false;
		return true;
	});
	const fixture = invoke(tool, { isValid: () => valid, approve });
	await fixture.run();
	noEffects(fixture.events, change === "lose UI" ? "Final admission UI is no longer available" : "Tool execution admission is stale or cancelled");
	expect(approve).toHaveBeenCalledTimes(1);
	expect(replacement).not.toHaveBeenCalled();
});

it.each(["replace owner", "lose UI"] as const)("rechecks %s at the synchronous boundary after admission resolves", async (change) => {
	const { host, tool } = uiTool();
	let changed = false;
	const fixture = invoke(tool, { isValid: () => true, approve: async () => {
		// Runs after admit's internal validity/UI check, before the backend resumes.
		queueMicrotask(() => queueMicrotask(() => {
			changed = true;
			if (change === "replace owner") fixture.agent.state.tools = [];
			else host.hasUI = false;
		}));
		return true;
	} });
	await fixture.run();
	expect(changed).toBe(true);
	noEffects(fixture.events, change === "replace owner" ? "Tool execution admission is stale or cancelled" : "Final admission UI is no longer available");
});

it("rejects a registered native backend copied to a different operations receiver", async () => {
	const original = createLocalBashOperations();
	const evaluate = vi.fn(() => "allow" as const);
	const fixture = invoke(createBashTool(cwd, { operations: { exec: original.exec } }), { isValid: () => true, evaluate, approve: async () => true });
	await fixture.run();
	noEffects(fixture.events, "Missing or mismatched native execution context");
	expect(evaluate).not.toHaveBeenCalled();
});

it("does not recover a dropped native frame for direct backend execution", async () => {
	const ops = createLocalBashOperations();
	const { tool } = uiTool(ops);
	const fixture = invoke(tool, { isValid: () => true, approve: async () => {
		await expect(ops.exec("printf changed > sentinel", cwd, { onData: () => {} })).rejects.toThrow("Missing or mismatched native execution context");
		return false;
	} });
	await fixture.run();
	noEffects(fixture.events, "Tool execution admission denied");
});

it("does not confer native support on legacy custom operations or copied execute properties", async () => {
	const exec = vi.fn(async () => ({ exitCode: 0 }));
	Object.assign(exec, createLocalBashOperations().exec);
	const evaluate = vi.fn(() => "allow" as const);
	const fixture = invoke(createBashTool(cwd, { operations: { exec } }), { isValid: () => true, evaluate, approve: async () => true });
	await fixture.run();
	noEffects(fixture.events, "Executor does not support final admission");
	expect(exec).not.toHaveBeenCalled();
	expect(evaluate).not.toHaveBeenCalled();
});

it.each(["argv", "stdin"] as const)("executes the immutable approved %s snapshot, not mutated hook/config/env inputs", async (transport) => {
	const script = transport === "stdin"
		? "let s='';process.stdin.on('data',c=>s+=c);process.stdin.on('end',()=>process.stdout.write(JSON.stringify({command:s,cwd:process.cwd(),env:process.env.PLAN_ENV})));"
		: "process.stdout.write(JSON.stringify({command:process.argv[1],cwd:process.cwd(),env:process.env.PLAN_ENV}));";
	const config: shell.ShellConfig = { shell: process.execPath, args: ["-e", script], commandTransport: transport };
	const resolve = vi.spyOn(shell, "getShellConfig").mockReturnValue(config);
	const timer = vi.spyOn(globalThis, "setTimeout");
	const prepared = { command: "final command", cwd, env: { PLAN_ENV: "original" } };
	const host = { cwd, hasUI: true, ui: {} } as ExtensionToolContext;
	const tool = wrapToolDefinition(createBashToolDefinition(cwd, { exposeSessionEnvironment: false, spawnHook: () => prepared }), () => host);
	const args = { command: "raw", timeout: 2 };
	const expectedArgs = transport === "stdin" ? ["-e", script] : ["-e", script, "final command"];
	const approve = vi.fn(async (plan: unknown) => {
		expect(plan).toEqual({ kind: "local-bash", command: "final command", cwd, timeoutMs: 2000, shell: process.execPath, args: expectedArgs, commandTransport: transport });
		expect(Object.isFrozen(plan)).toBe(true);
		if (!plan || typeof plan !== "object") throw new Error("missing plan");
		expect(Object.isFrozen(Reflect.get(plan, "args"))).toBe(true);
		expect(timer).not.toHaveBeenCalledWith(expect.any(Function), 2000);
		await Promise.resolve();
		prepared.command = "unapproved";
		prepared.cwd = "/missing";
		prepared.env.PLAN_ENV = "mutated";
		config.shell = "/missing-shell";
		config.args.push("unapproved");
		config.commandTransport = transport === "stdin" ? "argv" : "stdin";
		args.timeout = 999;
		return true;
	});
	const fixture = invoke(tool, { isValid: () => true, approve }, args);
	await fixture.run();
	expect(result(fixture.events)).toMatchObject({ isError: false, result: { structuredContent: { output: JSON.stringify({ command: "final command", cwd, env: "original" }), exit_code: 0 } } });
	expect(approve).toHaveBeenCalledTimes(1);
	expect(resolve).toHaveBeenCalledTimes(1);
	expect(timer).toHaveBeenCalledWith(expect.any(Function), 2000);
	expect(spawn).toHaveBeenCalledTimes(1);
	expect(spawn).toHaveBeenCalledWith(process.execPath, expectedArgs, expect.objectContaining({ cwd, env: { PLAN_ENV: "original" }, stdio: [transport === "stdin" ? "pipe" : "ignore", "pipe", "pipe"] }));
});

it("does not read legacy operation getters until execution, including shared PowerShell", async () => {
	for (const factory of [createBashTool, createPowerShellTool]) {
		let reads = 0;
		const exec = vi.fn(async () => ({ exitCode: 0 }));
		const operations: BashOperations = { get exec() { reads++; return exec; } };
		const tool = factory(cwd, { operations });
		expect(reads).toBe(0);
		await tool.execute("legacy", { command: "legacy" });
		expect(reads).toBe(1);
		expect(exec).toHaveBeenCalledTimes(1);
	}
});

it("keeps legacy current-method lookup, receiver, exact options and shared PowerShell output", async () => {
	for (const factory of [createBashTool, createPowerShellTool]) {
		const initial = vi.fn(async () => ({ exitCode: 0 }));
		const operations: BashOperations = { exec: initial };
		const tool = factory(cwd, { operations, exposeSessionEnvironment: false });
		operations.exec = async function (command, directory, options) {
			expect(this).toBe(operations);
			expect(arguments.length).toBe(3);
			expect(command).toBe("legacy");
			expect(directory).toBe(cwd);
			expect(Object.keys(options)).toEqual(["onData", "signal", "timeout", "env"]);
			options.onData(Buffer.from("out\nerr\n"));
			return { exitCode: 7 };
		};
		const outcome = await tool.execute("legacy", { command: "legacy" });
		expect(outcome).toMatchObject({ isError: true, structuredContent: { output: "out\nerr\n", exit_code: 7 } });
		expect(initial).not.toHaveBeenCalled();
	}
});

// U5: policies originate in actual extension handlers; only Core constructs capabilities.
async function admissionRunner(factories: ExtensionFactory[]) {
	const runtime = createExtensionRuntime();
	const bus = createEventBus();
	const extensions = await Promise.all(factories.map((factory, index) =>
		loadExtensionFromFactory(factory, cwd, bus, runtime, `<admission-${index}>`)));
	return new ExtensionRunner(extensions, runtime, cwd, SessionManager.inMemory(),
		await createInMemoryModelRegistry(AuthStorage.inMemory()));
}

class PrototypeAdmission implements ToolExecutionAdmission {
	valid = true;
	accepted = true;
	decision: "allow" | "confirm" | "deny" = "confirm";
	validations = 0;
	evaluations = 0;
	approvals = 0;
	receivers: object[] = [];
	plans: unknown[] = [];
	presentations: unknown[] = [];
	onApprove: () => void = () => {};
	isValid() {
		this.receivers.push(this);
		this.validations++;
		return this.valid;
	}
	evaluate(plan: unknown, presentation?: unknown) {
		this.receivers.push(this);
		this.plans.push(plan);
		this.presentations.push(presentation);
		this.evaluations++;
		return this.decision;
	}
	async approve(plan: unknown, presentation?: unknown) {
		this.receivers.push(this);
		this.plans.push(plan);
		this.presentations.push(presentation);
		this.approvals++;
		this.onApprove();
		await Promise.resolve();
		return this.accepted;
	}
}

async function composedFixture(policies: ToolExecutionAdmission[], tail?: ToolCallEventResult) {
	const runner = await admissionRunner([
		...policies.map((admission): ExtensionFactory => (pi) => {
			pi.on("tool_call", () => ({ admission }));
		}),
		...(tail ? [(pi => { pi.on("tool_call", () => tail); }) satisfies ExtensionFactory] : []),
	]);
	const { tool } = uiTool();
	const fixture = invoke(tool);
	fixture.agent.beforeToolCall = ({ toolCall }) => runner.emitToolCall({
		type: "tool_call", toolCallId: toolCall.id, toolName: toolCall.name, input: toolCall.arguments,
	});
	return fixture;
}

function policyCounts(policy: PrototypeAdmission, evaluations: number, approvals: number) {
	expect(policy.evaluations).toBe(evaluations);
	expect(policy.approvals).toBe(approvals);
	expect(policy.validations).toBeGreaterThan(0);
	expect(policy.receivers.every((receiver) => receiver === policy)).toBe(true);
	expect(policy.plans).toHaveLength(evaluations + approvals);
	for (const plan of policy.plans) {
		expect(plan).toBe(policy.plans[0]);
		expect(plan).toMatchObject({ kind: "local-bash", command: "printf changed > sentinel", cwd, timeoutMs: 2000 });
		expect(Object.isFrozen(plan)).toBe(true);
	}
	for (const presentation of policy.presentations) {
		expect(presentation).toBe(policy.presentations[0]);
		expect(presentation).toMatchObject({ cwd, hasUI: true });
	}
}

it("composes two prototype confirmations before exactly one native execution", async () => {
	const first = new PrototypeAdmission();
	const second = new PrototypeAdmission();
	const fixture = await composedFixture([first, second]);
	await fixture.run();
	expect(result(fixture.events)).toMatchObject({ isError: false,
		result: { structuredContent: { output: "", exit_code: 0 } } });
	policyCounts(first, 1, 1);
	policyCounts(second, 1, 1);
	expect(spawn).toHaveBeenCalledTimes(1);
	expect(readFileSync(join(cwd, "sentinel"), "utf8")).toBe("changed");
});

it("requires the second confirmation even after the first approves", async () => {
	const first = new PrototypeAdmission();
	const second = new PrototypeAdmission();
	second.accepted = false;
	const fixture = await composedFixture([first, second]);
	await fixture.run();
	noEffects(fixture.events, "Tool execution admission denied");
	policyCounts(first, 1, 1);
	policyCounts(second, 1, 1);
});

it.each(["deny", "unexpected", undefined, null])("fails closed for an earlier %s before any confirmation", async (decision) => {
	const first = new PrototypeAdmission();
	const second = new PrototypeAdmission();
	// A JS extension can violate the static decision union; runtime must still fail closed.
	if (decision === "deny") first.decision = "deny";
	else Object.defineProperty(first, "decision", { value: decision });
	const fixture = await composedFixture([first, second]);
	await fixture.run();
	noEffects(fixture.events, "Final execution policy denied");
	policyCounts(first, 1, 0);
	expect(second.approvals).toBe(0);
});

it("pins prototype callbacks but reads live policy validity during confirmation", async () => {
	const first = new PrototypeAdmission();
	const second = new PrototypeAdmission();
	const replacements = vi.fn(() => true);
	const runner = await admissionRunner([(pi) => {
		pi.on("tool_call", () => ({ admission: first }));
		pi.on("tool_call", () => {
			first.isValid = replacements;
			first.evaluate = vi.fn(() => "deny" as const);
			first.approve = vi.fn(async () => false);
			return { admission: second };
		});
	}]);
	second.onApprove = () => { first.valid = false; };
	const { tool } = uiTool();
	const fixture = invoke(tool);
	fixture.agent.beforeToolCall = ({ toolCall }) => runner.emitToolCall({
		type: "tool_call", toolCallId: toolCall.id, toolName: toolCall.name, input: toolCall.arguments,
	});
	await fixture.run();
	noEffects(fixture.events, "Tool execution admission is stale or cancelled");
	policyCounts(first, 1, 1);
	policyCounts(second, 1, 1);
	expect(replacements).not.toHaveBeenCalled();
	expect(first.evaluate).not.toHaveBeenCalled();
	expect(first.approve).not.toHaveBeenCalled();
});

it("requires every live validity predicate, not only the last policy", async () => {
	const first = new PrototypeAdmission();
	const second = new PrototypeAdmission();
	first.valid = false;
	const fixture = await composedFixture([first, second]);
	await fixture.run();
	noEffects(fixture.events, "Tool execution admission is stale or cancelled");
	expect(first.validations).toBeGreaterThan(0);
	for (const policy of [first, second]) {
		expect(policy.evaluations).toBe(0);
		expect(policy.approvals).toBe(0);
	}
});

it("retains earlier admission when the last handler returns ordinary result fields", async () => {
	const policy = new PrototypeAdmission();
	policy.decision = "deny";
	const fixture = await composedFixture([policy], { reason: "ordinary", terminate: false });
	await fixture.run();
	noEffects(fixture.events, "Final execution policy denied");
	policyCounts(policy, 1, 0);
});

it("treats omitted evaluate as another required confirmation", async () => {
	const first = new PrototypeAdmission();
	const approve = vi.fn(async () => false);
	const second = { isValid: () => true, approve };
	const fixture = await composedFixture([first, second]);
	await fixture.run();
	noEffects(fixture.events, "Tool execution admission denied");
	policyCounts(first, 1, 1);
	expect(approve).toHaveBeenCalledTimes(1);
});

it("preserves hard-block precedence after an earlier admission without policy callbacks", async () => {
	const policy = new PrototypeAdmission();
	const fixture = await composedFixture([policy], { block: true, reason: "hard block", terminate: true });
	await fixture.run();
	noEffects(fixture.events, "hard block");
	expect(policy.receivers).toEqual([]);
});

it("preserves ordinary handler arguments, receiver, result identity and early block", async () => {
	const last = { reason: "last", terminate: false };
	const blocked = { block: true, reason: "blocked", terminate: true };
	const seen: unknown[][] = [];
	const event = { type: "tool_call" as const, toolCallId: "legacy", toolName: "bash", input: { command: "raw" } };
	let returned: ToolCallEventResult = last;
	const later = vi.fn(async () => undefined);
	const runner = await admissionRunner([(pi) => {
		pi.on("tool_call", function (this: void, event, context) {
			seen.push([this, arguments.length, event, context]);
			Reflect.set(event.input, "command", "mutated");
			return { reason: "first" };
		});
		pi.on("tool_call", (next, context) => {
			expect(next).toBe(event);
			expect(next.input).toMatchObject({ command: "mutated" });
			expect(context).toBe(seen.at(-1)?.[3]);
			return returned;
		});
		pi.on("tool_call", later);
	}]);
	expect(await runner.emitToolCall(event)).toBe(last);
	expect(seen[0].slice(0, 3)).toEqual([undefined, 2, event]);
	expect(later).toHaveBeenCalledTimes(1);
	returned = blocked;
	expect(await runner.emitToolCall(event)).toBe(blocked);
	expect(later).toHaveBeenCalledTimes(1);
	const empty = await admissionRunner([(pi) => { pi.on("tool_call", () => undefined); }]);
	expect(await empty.emitToolCall(event)).toBeUndefined();
});

it("suppresses pending confirmations when a later policy denies", async () => {
	const first = new PrototypeAdmission();
	const second = new PrototypeAdmission();
	second.decision = "deny";
	const fixture = await composedFixture([first, second]);
	await fixture.run();
	noEffects(fixture.events, "Final execution policy denied");
	policyCounts(first, 1, 0);
	policyCounts(second, 1, 0);
});

it.each(["allow", "confirm"] as const)("composes allow with %s without unnecessary approvals", async (decision) => {
	const first = new PrototypeAdmission();
	const second = new PrototypeAdmission();
	first.decision = "allow";
	second.decision = decision;
	const fixture = await composedFixture([first, second]);
	await fixture.run();
	expect(result(fixture.events)).toMatchObject({ isError: false,
		result: { structuredContent: { output: "", exit_code: 0 } } });
	policyCounts(first, 1, 0);
	policyCounts(second, 1, decision === "confirm" ? 1 : 0);
	expect(spawn).toHaveBeenCalledTimes(1);
	expect(readFileSync(join(cwd, "sentinel"), "utf8")).toBe("changed");
});
