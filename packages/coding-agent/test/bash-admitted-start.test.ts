import type * as ChildProcess from "node:child_process";
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentEvent, type AgentTool, bindToolExecutionContext, type ToolExecutionAdmission, type ToolExecutionContext } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, createAssistantMessageEventStream } from "@earendil-works/pi-ai";
import { afterEach, expect, it, vi } from "vitest";
import { type BashOperations, createBashTool } from "../src/core/tools/bash.ts";
import * as shell from "../src/utils/shell.ts";

vi.mock("child_process", async (importOriginal) => {
	const actual = await importOriginal<typeof ChildProcess>();
	return { ...actual, spawn: vi.fn(actual.spawn) };
});

afterEach(() => {
	vi.restoreAllMocks();
	vi.mocked(spawn).mockClear();
});

const allow: ToolExecutionAdmission = {
	isValid: () => true,
	evaluate: () => "allow",
	approve: async () => false,
};

function invoke(
	tool: AgentTool,
	command: string,
	admission = allow,
	options?: { timeout?: number; onAgent?: (agent: Agent) => void },
) {
	const events: AgentEvent[] = [];
	const agent = new Agent({
		initialState: { tools: [tool] },
		beforeToolCall: async () => ({ admission }),
		streamFn: () => {
			const stream = createAssistantMessageEventStream();
			const message: AssistantMessage = {
				role: "assistant", content: [{ type: "toolCall", id: "start", name: tool.name, arguments: { command, timeout: options?.timeout ?? 2 } }],
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
	options?.onAgent?.(agent);
	return agent.prompt("synthetic start fixture").then(() => {
		const end = events.find((event) => event.type === "tool_execution_end");
		if (!end || end.type !== "tool_execution_end") throw new Error("missing tool result");
		return end;
	});
}

function consumer(cwd: string, onStarted: () => void | Promise<void>, shellPath = "/bin/sh") {
	const options = { onStarted, shellPath, exposeSessionEnvironment: false };
	const native = createBashTool(cwd, options);
	const delegate = native.execute;
	const tool: typeof native = {
		...native,
		execute: (id, args, signal?, update?, context?: ToolExecutionContext) => native.execute(id, args, signal, update, context),
	};
	bindToolExecutionContext(tool.execute, delegate, () => native.execute === delegate);
	return tool;
}

it("acknowledges exactly once after real spawn, never during final admission", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "ack-"));
	let starts = 0;
	let spawned = false;
	const realSpawn = vi.mocked(spawn).getMockImplementation()!;
	vi.mocked(spawn).mockImplementationOnce((...args: Parameters<typeof spawn>) => {
		const child = realSpawn(...args);
		child.once("spawn", () => { spawned = true; });
		return child;
	});
	const tool = consumer(cwd, function (this: unknown, ...args: unknown[]) {
		expect(this).toBeUndefined();
		expect(args).toEqual([]);
		expect(spawned).toBe(true);
		starts++;
	});
	const result = await invoke(tool, "printf actual; printf error >&2; exit 7", {
		...allow,
		evaluate: (plan) => {
			expect(starts).toBe(0);
			expect(spawn).not.toHaveBeenCalled();
			expect(Object.isFrozen(plan)).toBe(true);
			return "allow";
		},
	});
	expect(starts).toBe(1);
	expect(result.result.structuredContent).toMatchObject({ output: "actualerror", exit_code: 7 });
});

it("does not acknowledge admission denial", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "deny-ack-"));
	const started = vi.fn();
	const result = await invoke(consumer(cwd, started), "printf forbidden", { ...allow, evaluate: () => "deny" });
	expect(result.isError).toBe(true);
	expect(spawn).not.toHaveBeenCalled();
	expect(started).not.toHaveBeenCalled();
});

it("does not acknowledge a real OS startup error", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "failed-ack-"));
	const shellPath = join(cwd, "not-executable");
	writeFileSync(shellPath, "#!/bin/sh\nexit 0\n", { mode: 0o600 });
	const started = vi.fn();
	const result = await invoke(consumer(cwd, started, shellPath), "printf forbidden");
	expect(result).toMatchObject({ isError: true, result: { content: [{ type: "text", text: expect.stringContaining("EACCES") }] } });
	expect(started).not.toHaveBeenCalled();
});

it.each([false, true])("fails completion and stops its still-live child on callback failure (async=%s)", async (asyncFailure) => {
	const cwd = mkdtempSync(join(tmpdir(), "failed-callback-"));
	const effect = join(cwd, "delayed-effect");
	let starts = 0;
	const result = await invoke(consumer(cwd, () => {
		starts++;
		if (asyncFailure) return Promise.reject(new Error("publication failed"));
		throw new Error("publication failed");
	}), `sleep 0.2; printf orphan > ${JSON.stringify(effect)}`);
	expect(starts).toBe(1);
	expect(result).toMatchObject({ isError: true, result: { content: [{ type: "text", text: "publication failed" }] } });
	await new Promise((resolve) => setTimeout(resolve, 300));
	expect(existsSync(effect)).toBe(false);
});

it("awaits pending callback rejection after fast exit without signalling the exited child", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "fast-ack-"));
	let reject!: (error: Error) => void;
	const notification = new Promise<void>((_resolve, fail) => { reject = fail; });
	let closed!: () => void;
	const termination = new Promise<void>((resolve) => { closed = resolve; });
	const realSpawn = vi.mocked(spawn).getMockImplementation()!;
	vi.mocked(spawn).mockImplementationOnce((...args: Parameters<typeof spawn>) => {
		const child = realSpawn(...args);
		child.once("close", closed);
		return child;
	});
	// Never send a signal in this fast-exit test, even on a regression.
	const kill = vi.spyOn(shell, "killProcessTree").mockImplementation(() => {});
	let starts = 0;
	let completed = false;
	const completion = invoke(consumer(cwd, () => { starts++; return notification; }), "exit 0")
		.then((result) => { completed = true; return result; });
	await termination;
	await new Promise((resolve) => setTimeout(resolve, 20));
	const premature = completed;
	// Attach a handler for baseline RED, where the missing hook never consumes it.
	void notification.catch(() => {});
	reject(new Error("late publication failed"));
	const result = await completion;
	expect(starts).toBe(1);
	expect(premature).toBe(false);
	expect(kill).not.toHaveBeenCalled();
	expect(result).toMatchObject({ isError: true, result: { content: [{ type: "text", text: "late publication failed" }] } });
});

it.each(["abort", "timeout"] as const)("does not extend child signal authority into a pending callback after completion: %s", async (mode) => {
	const cwd = mkdtempSync(join(tmpdir(), "post-completion-ack-"));
	let settle!: () => void;
	const notification = new Promise<void>((resolve) => { settle = resolve; });
	let closed!: () => void;
	const termination = new Promise<void>((resolve) => { closed = resolve; });
	let child: ChildProcess.ChildProcess | undefined;
	const realSpawn = vi.mocked(spawn).getMockImplementation()!;
	vi.mocked(spawn).mockImplementationOnce((...args: Parameters<typeof spawn>) => {
		child = realSpawn(...args);
		child.once("close", closed);
		return child;
	});
	// Observe the actual OS-signal boundary. Intercept only this known-exited
	// owned child/group; never send stale or unrelated signals, even on RED.
	const kill = vi.spyOn(process, "kill").mockImplementation((pid) => {
		if (child?.pid && Math.abs(pid) === child.pid && child.exitCode !== null) return true;
		throw new Error("Unexpected signal target in owned fast-exit fixture");
	});
	let agent: Agent | undefined;
	let starts = 0;
	let completed = false;
	const completion = invoke(consumer(cwd, () => { starts++; return notification; }), "printf done", allow, {
		timeout: mode === "timeout" ? 0.3 : 2,
		onAgent: (value) => { agent = value; },
	}).then((result) => { completed = true; return result; });
	await termination;
	// Let waitForChildProcess and its await continuation finish before the
	// external event; this tests only the newly introduced callback interval.
	await new Promise((resolve) => setTimeout(resolve, 20));
	const pending = !completed;
	if (mode === "abort") agent!.abort();
	else await new Promise((resolve) => setTimeout(resolve, 350));
	const signals = kill.mock.calls.map(([pid, signal]) => ({ pid, signal }));
	settle();
	const result = await completion;
	expect(child?.exitCode).toBe(0);
	expect(starts).toBe(1);
	expect(pending).toBe(true);
	expect(signals).toEqual([]);
	if (mode === "abort") expect(result.isError).toBe(true);
	else expect(result).toMatchObject({ isError: false, result: { structuredContent: { output: "done", exit_code: 0 } } });
});

it("forwards the original host callback identity and preserves its no-receiver signature", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "callback-contract-"));
	let receiver: unknown = "not called";
	let count = -1;
	function onStarted(this: unknown, ...args: unknown[]) { receiver = this; count = args.length; }
	const exec = vi.fn<BashOperations["exec"]>(async (_command, _cwd, options) => {
		expect(options.onStarted).toBe(onStarted);
		const callback = options.onStarted;
		await callback?.();
		return { exitCode: 0 };
	});
	await createBashTool(cwd, { operations: { exec }, onStarted, exposeSessionEnvironment: false })
		.execute("legacy", { command: "unused" });
	expect(receiver).toBeUndefined();
	expect(count).toBe(0);
});

it("does not add optional keys or read legacy executor getters until execution when the hook is absent", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "legacy-ack-"));
	let reads = 0;
	const calls: string[][] = [];
	const operations: BashOperations = {
		get exec() {
			reads++;
			const exec: BashOperations["exec"] = async function (this: BashOperations, _command, _cwd, options) {
				expect(this).toBe(operations);
				calls.push(Object.keys(options));
				return { exitCode: 0 };
			};
			return exec;
		},
	};
	const tool = createBashTool(cwd, { operations, exposeSessionEnvironment: false });
	expect(reads).toBe(0);
	await tool.execute("legacy", { command: "unused" });
	expect(reads).toBe(1);
	expect(calls).toEqual([["onData", "signal", "timeout", "env"]]);
});

it("requireAdmission rejects a direct contextless call before hooks, spawn, acknowledgement or stored effects", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "required-admission-"));
	const effect = join(cwd, "forbidden-effect");
	const before = readdirSync(cwd);
	const started = vi.fn();
	let hooks = 0;
	const tool = createBashTool(cwd, {
		requireAdmission: true,
		shellPath: "/bin/sh",
		exposeSessionEnvironment: false,
		onStarted: started,
		spawnHook: (context) => { hooks++; return context; },
	});
	let result: unknown;
	let error: unknown;
	try {
		result = await tool.execute("direct", {
			command: `printf forbidden > ${JSON.stringify(effect)}; printf unexpected; printf unexpected-error >&2; exit 7`,
		});
	} catch (caught) {
		error = caught;
	}
	expect({
		error: error instanceof Error ? error.message : error,
		result,
		hooks,
		spawns: vi.mocked(spawn).mock.calls.length,
		acknowledgements: started.mock.calls,
		storedFiles: readdirSync(cwd),
	}).toEqual({
		error: "Host admission context required",
		result: undefined,
		hooks: 0,
		spawns: 0,
		acknowledgements: [],
		storedFiles: before,
	});
});

it("requireAdmission accepts a real Core-admitted execution and acknowledges its actual child", async () => {
	const cwd = mkdtempSync(join(tmpdir(), "required-core-admission-"));
	const started = vi.fn();
	let hooks = 0;
	let evaluations = 0;
	const tool = createBashTool(cwd, {
		requireAdmission: true,
		shellPath: "/bin/sh",
		exposeSessionEnvironment: false,
		onStarted: started,
		spawnHook: (context) => { hooks++; return context; },
	});
	const result = await invoke(tool, "printf admitted", {
		...allow,
		evaluate: () => {
			evaluations++;
			expect(started).not.toHaveBeenCalled();
			expect(spawn).not.toHaveBeenCalled();
			return "allow";
		},
	});
	expect(result).toMatchObject({
		isError: false,
		result: {
			content: [{ type: "text", text: "admitted" }],
			structuredContent: { output: "admitted", exit_code: 0, truncated: false },
		},
	});
	expect(evaluations).toBe(1);
	expect(hooks).toBe(1);
	expect(spawn).toHaveBeenCalledTimes(1);
	expect(started).toHaveBeenCalledTimes(1);
	expect(readdirSync(cwd)).toEqual([]);
});

it.each([undefined, false])("requireAdmission=%s preserves the direct legacy execution contract", async (requireAdmission) => {
	const cwd = mkdtempSync(join(tmpdir(), "legacy-direct-admission-"));
	const started = vi.fn();
	let hooks = 0;
	const tool = createBashTool(cwd, {
		...(requireAdmission === undefined ? {} : { requireAdmission }),
		shellPath: "/bin/sh",
		exposeSessionEnvironment: false,
		onStarted: started,
		spawnHook: (context) => { hooks++; return context; },
	});
	const result = await tool.execute("legacy-direct", { command: "printf legacy" });
	expect(result.content).toEqual([{ type: "text", text: "legacy" }]);
	expect(result.structuredContent).toMatchObject({ output: "legacy", exit_code: 0, truncated: false });
	expect(result).not.toMatchObject({ isError: true });
	expect(hooks).toBe(1);
	expect(spawn).toHaveBeenCalledTimes(1);
	expect(started).toHaveBeenCalledTimes(1);
	expect(readdirSync(cwd)).toEqual([]);
});
