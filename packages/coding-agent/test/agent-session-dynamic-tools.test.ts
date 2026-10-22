import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type AgentTool, registerToolExecutionContext, type ToolExecutionContext } from "@earendil-works/pi-agent-core";
import { getModel } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createToolExecutionContext } from "../../agent/src/tool-execution-context.ts";
import type { ExtensionRunner } from "../src/core/extensions/runner.ts";
import type { ExtensionToolContext, ToolDefinition } from "../src/core/extensions/types.ts";
import { wrapRegisteredTool } from "../src/core/extensions/wrapper.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createBashTool } from "../src/core/tools/bash.ts";
import { createPowerShellTool } from "../src/core/tools/powershell.ts";
import {
	createToolDefinitionFromAgentTool,
	wrapToolDefinition,
	wrapToolDefinitions,
} from "../src/core/tools/tool-definition-wrapper.ts";

describe("native tool definition bridges", () => {
	const parameters = Type.Object({});
	const result = { content: [], details: undefined };
	const allow = { isValid: () => true, evaluate: () => "allow" as const, approve: async () => false };

	it("preserves legacy core receiver and exactly four arguments", async () => {
		const tool: AgentTool<typeof parameters> = {
			name: "legacy",
			label: "legacy",
			description: "legacy",
			parameters,
			async execute() {
				expect(this).toBe(tool);
				expect(arguments.length).toBe(4);
				return result;
			},
		};
		const definition = createToolDefinitionFromAgentTool(tool);
		await wrapToolDefinition(definition).execute("legacy", {});
	});

	it("preserves extension argument five and lazy current context getters", async () => {
		let factoryCalls = 0;
		let reads = 0;
		let hasUI = false;
		// Ordinary extension-context fixture, never an admission capability.
		const extensionContext = {
			get hasUI() {
				reads++;
				return hasUI;
			},
		} as ExtensionToolContext;
		const seen: boolean[] = [];
		const definition: ToolDefinition<typeof parameters> = {
			name: "extension",
			label: "extension",
			description: "extension",
			parameters,
			async execute(_id, _params, _signal, _update, ctx) {
				expect(this).toBe(definition);
				expect(arguments.length).toBe(5);
				expect(ctx).toBe(extensionContext);
				seen.push(ctx.hasUI);
				return result;
			},
		};
		const wrapped = wrapToolDefinition(definition, () => {
			factoryCalls++;
			return extensionContext;
		});
		expect(factoryCalls).toBe(0);
		expect(reads).toBe(0);
		await wrapped.execute("first", {});
		hasUI = true;
		await wrapped.execute("second", {}, undefined, undefined, extensionContext);
		expect(factoryCalls).toBe(1);
		expect(seen).toEqual([false, true]);
	});

	it("binds definitions to the real executor and forwards native context as argument six", async () => {
		let effects = 0;
		let received: ToolExecutionContext | undefined;
		const definition: ToolDefinition<typeof parameters> = {
			name: "native",
			label: "native",
			description: "native",
			parameters,
			async execute(_id, _params, _signal, _update, ctx, nativeContext) {
				expect(this).toBe(definition);
				expect(ctx).toBeUndefined();
				nativeContext?.assertValid(definition.execute);
				received = nativeContext;
				await nativeContext?.admit({ command: "final" });
				effects++;
				return result;
			},
		};
		const execute = definition.execute;
		registerToolExecutionContext(execute, () => definition.execute === execute);
		const [wrapped] = wrapToolDefinitions([definition]);
		const context = createToolExecutionContext(wrapped.execute, allow, () => true);
		await wrapped.execute("native", {}, undefined, undefined, context);
		expect(received).toBe(context);
		context.finish();
		context.close();
		await expect(wrapped.execute("closed", {}, undefined, undefined, context)).rejects.toThrow();
		expect(effects).toBe(1);
	});

	it("round-trips native core tools with pinned delegates and fails closed on replacement", async () => {
		let effects = 0;
		const tool: AgentTool<typeof parameters> = {
			name: "core",
			label: "core",
			description: "core",
			parameters,
			async execute(id, _params, _signal, _update, context?: ToolExecutionContext) {
				expect(this).toBe(tool);
				if (id === "legacy") {
					expect(arguments.length).toBe(4);
					return result;
				}
				expect(context).toBeDefined();
				context?.assertValid(tool.execute);
				await context?.admit({ command: "core" });
				effects++;
				return result;
			},
		};
		const execute = tool.execute;
		registerToolExecutionContext(execute, () => tool.execute === execute);
		const definition = createToolDefinitionFromAgentTool(tool);
		const wrapped = wrapToolDefinition(definition);
		const context = createToolExecutionContext(wrapped.execute, allow, () => true);
		await wrapped.execute("native", {}, undefined, undefined, context);
		context.finish();
		context.close();
		const stale = createToolExecutionContext(wrapped.execute, allow, () => true);
		tool.execute = async function (this: AgentTool<typeof parameters>) {
			expect(this).toBe(tool);
			expect(arguments.length).toBe(4);
			effects += 100;
			return result;
		};
		await expect(wrapped.execute("stale", {}, undefined, undefined, stale)).rejects.toThrow();
		expect(effects).toBe(1);
		stale.close();
		// Legacy calls resolve the current executor, with its owner and four arguments.
		await wrapped.execute("legacy", {});
		expect(effects).toBe(101);
		await expect(wrapped.execute("stale-after-legacy", {}, undefined, undefined, stale)).rejects.toThrow();
		expect(effects).toBe(101);
	});

	it("preserves denying policy receivers through the bridge with no effects or approval", async () => {
		let effects = 0;
		const definition: ToolDefinition<typeof parameters> = {
			name: "denied",
			label: "denied",
			description: "denied",
			parameters,
			async execute(_id, _params, _signal, _update, _ctx, context) {
				await context?.admit("final plan");
				effects++;
				return result;
			},
		};
		class Policy {
			decision = "deny" as const;
			approvals = 0;
			isValid() {
				return this.approvals === 0;
			}
			evaluate() {
				return this.decision;
			}
			async approve() {
				this.approvals++;
				return true;
			}
		}
		const execute = definition.execute;
		registerToolExecutionContext(execute, () => definition.execute === execute);
		const wrapped = wrapToolDefinition(definition);
		const policy = new Policy();
		const context = createToolExecutionContext(wrapped.execute, policy, () => true);
		await expect(wrapped.execute("denied", {}, undefined, undefined, context)).rejects.toThrow("denied");
		expect(effects).toBe(0);
		expect(policy.approvals).toBe(0);
		context.close();
	});

	it("rejects definition replacement during approval and does not call the replacement", async () => {
		let effects = 0;
		let release!: (approved: boolean) => void;
		let started!: () => void;
		const pending = new Promise<boolean>((resolve) => {
			release = resolve;
		});
		const approving = new Promise<void>((resolve) => {
			started = resolve;
		});
		const definition: ToolDefinition<typeof parameters> = {
			name: "waiting",
			label: "waiting",
			description: "waiting",
			parameters,
			async execute(_id, _params, _signal, _update, _ctx, context) {
				await context?.admit("waiting");
				effects++;
				return result;
			},
		};
		const execute = definition.execute;
		registerToolExecutionContext(execute, () => definition.execute === execute);
		const wrapped = wrapToolDefinition(definition);
		const context = createToolExecutionContext(wrapped.execute, {
			isValid: () => true,
			approve: async () => {
				started();
				return pending;
			},
		}, () => true);
		const execution = wrapped.execute("waiting", {}, undefined, undefined, context);
		await approving;
		definition.execute = async function (this: ToolDefinition<typeof parameters>) {
			expect(this).toBe(definition);
			expect(arguments.length).toBe(5);
			effects += 100;
			return result;
		};
		release(true);
		await expect(execution).rejects.toThrow("stale");
		expect(effects).toBe(0);
		context.close();
		// Legacy calls resolve the current executor, with its owner and five arguments.
		await wrapped.execute("legacy", {});
		expect(effects).toBe(100);
		await expect(wrapped.execute("stale-after-legacy", {}, undefined, undefined, context)).rejects.toThrow();
		expect(effects).toBe(100);
	});

	it("does not acquire native capability by copying executor properties", () => {
		const definition: ToolDefinition<typeof parameters> = {
			name: "copy",
			label: "copy",
			description: "copy",
			parameters,
			async execute() {
				return result;
			},
		};
		const execute = definition.execute;
		registerToolExecutionContext(execute, () => definition.execute === execute);
		const copied = Object.assign(async () => result, execute);
		const wrapped = wrapToolDefinition({ ...definition, execute: copied });
		expect(() => createToolExecutionContext(wrapped.execute, allow, () => true)).toThrow();
	});

	it("forwards registered extension native context without eagerly creating UI context", async () => {
		let factories = 0;
		const extensionContext = { hasUI: false } as ExtensionToolContext;
		const definition: ToolDefinition<typeof parameters> = {
			name: "registered",
			label: "registered",
			description: "registered",
			parameters,
			async execute(_id, _params, _signal, _update, ctx, context) {
				expect(this).toBe(definition);
				expect(ctx).toBe(extensionContext);
				context?.assertValid(definition.execute);
				await context?.admit("registered");
				return result;
			},
		};
		const execute = definition.execute;
		registerToolExecutionContext(execute, () => definition.execute === execute);
		const runner: Pick<ExtensionRunner, "createToolContext"> = {
			createToolContext() {
				expect(this).toBe(runner);
				factories++;
				return extensionContext;
			},
		};
		const wrapped = wrapRegisteredTool({
			definition,
			sourceInfo: { path: "test", source: "inline", scope: "temporary", origin: "top-level" },
		}, runner);
		expect(factories).toBe(0);
		const context = createToolExecutionContext(wrapped.execute, allow, () => true);
		await wrapped.execute("registered", {}, undefined, undefined, context);
		context.finish();
		context.close();
		expect(factories).toBe(1);
	});

	it("transports native context through a directly registered factory wrapper", async () => {
		let effects = 0;
		let received: ToolExecutionContext | undefined;
		const extensionContext = { hasUI: false } as ExtensionToolContext;
		const definition: ToolDefinition<typeof parameters> = {
			name: "factory",
			label: "factory",
			description: "factory",
			parameters,
			async execute(id, _params, _signal, _update, ctx, context) {
				expect(this).toBe(definition);
				expect(ctx).toBe(extensionContext);
				if (id === "legacy") {
					expect(arguments.length).toBe(5);
					expect(context).toBeUndefined();
					return result;
				}
				expect(context).toBeDefined();
				context?.assertValid(definition.execute);
				received = context;
				await context?.admit("factory final plan");
				effects++;
				return result;
			},
		};
		const execute = definition.execute;
		registerToolExecutionContext(execute, () => definition.execute === execute);
		// The same public wrapping factory used by built-in tool factories. Register
		// its result directly, without createToolDefinitionFromAgentTool().
		const factoryTool = wrapToolDefinition(definition);
		const runner: Pick<ExtensionRunner, "createToolContext"> = {
			createToolContext() {
				return extensionContext;
			},
		};
		const registered = wrapRegisteredTool({
			definition: factoryTool,
			sourceInfo: { path: "factory", source: "inline", scope: "temporary", origin: "top-level" },
		}, runner);
		await registered.execute("legacy", {});
		const context = createToolExecutionContext(registered.execute, allow, () => true);
		await registered.execute("factory", {}, undefined, undefined, context);
		expect(received).toBe(context);
		expect(effects).toBe(1);
		context.finish();
		context.close();
	});

	it("preserves shared PowerShell custom operations without a PowerShell process", async () => {
		const calls: string[] = [];
		const tool = createPowerShellTool("/synthetic", {
			operations: {
				async exec(command, cwd, options) {
					calls.push(`${cwd}:${command}`);
					options.onData(Buffer.from("fixture output"));
					return { exitCode: 0 };
				},
			},
		});
		const output = await wrapToolDefinition(createToolDefinitionFromAgentTool(tool)).execute("ps", {
			command: "Write-Output fixture",
		});
		expect(calls).toEqual(["/synthetic:Write-Output fixture"]);
		expect(output.content).toEqual([{ type: "text", text: "fixture output" }]);
	});
});

describe("AgentSession dynamic tool registration", () => {
	let tempDir: string;
	let agentDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-dynamic-tool-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		agentDir = join(tempDir, "agent");
		mkdirSync(agentDir, { recursive: true });
	});

	afterEach(() => {
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	it("exposes session state before custom bash spawn hooks and supports opting out", async () => {
		const settingsManager = SettingsManager.create(tempDir, agentDir);
		const sessionManager = SessionManager.create(tempDir, join(agentDir, "sessions"), { id: "bash-env-test" });
		let sessionEnv: NodeJS.ProcessEnv | undefined;
		let optedOutEnv: NodeJS.ProcessEnv | undefined;
		const resourceLoader = new DefaultResourceLoader({
			cwd: tempDir,
			agentDir,
			settingsManager,
			extensionFactories: [
				(pi) => {
					pi.registerTool(
						createBashTool(tempDir, {
							spawnHook: (ctx) => {
								sessionEnv = ctx.env;
								return ctx;
							},
						}),
					);
					pi.registerTool({
						...createBashTool(tempDir, {
							exposeSessionEnvironment: false,
							spawnHook: (ctx) => {
								optedOutEnv = ctx.env;
								return ctx;
							},
						}),
						name: "bash_without_session_env",
						label: "bash without session env",
					});
				},
			],
		});
		await resourceLoader.reload();

		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir,
			model,
			thinkingLevel: "high",
			settingsManager,
			sessionManager,
			resourceLoader,
		});

		const bashTool = session.agent.state.tools.find((tool) => tool.name === "bash")!;
		expect(session.systemPrompt).toContain(
			"You can inspect PI_* environment variables for current model and session details.",
		);
		await bashTool.execute("bash-env", { command: "printf ok" });
		expect(sessionEnv).toMatchObject({
			PI_SESSION_ID: session.sessionId,
			PI_SESSION_FILE: session.sessionFile,
			PI_PROVIDER: model.provider,
			PI_MODEL: model.id,
			PI_REASONING_LEVEL: session.thinkingLevel,
		});

		const optedOutBashTool = session.agent.state.tools.find((tool) => tool.name === "bash_without_session_env")!;
		await optedOutBashTool.execute("bash-no-env", { command: "printf ok" });
		expect(optedOutEnv).not.toHaveProperty("PI_SESSION_ID");
		expect(optedOutEnv).not.toHaveProperty("PI_SESSION_FILE");
		expect(optedOutEnv).not.toHaveProperty("PI_PROVIDER");
		expect(optedOutEnv).not.toHaveProperty("PI_MODEL");
		expect(optedOutEnv).not.toHaveProperty("PI_REASONING_LEVEL");

		session.dispose();
	});

	it("retains native capability when a core-typed factory result is registered directly", async () => {
		let effects = 0;
		let received: ToolExecutionContext | undefined;
		const parameters = Type.Object({});
		const definition: ToolDefinition<typeof parameters> = {
			name: "native_factory",
			label: "native factory",
			description: "Synthetic native backend, no process effects",
			parameters,
			async execute(id, _params, _signal, _update, ctx, context) {
				expect(this).toBe(definition);
				expect(ctx.cwd).toBe(tempDir);
				if (id === "legacy") {
					expect(arguments.length).toBe(5);
					expect(context).toBeUndefined();
					return { content: [], details: undefined };
				}
				expect(context).toBeDefined();
				context?.assertValid(definition.execute);
				received = context;
				await context?.admit({ command: "synthetic final plan" });
				effects++;
				return { content: [], details: undefined };
			},
		};
		const execute = definition.execute;
		registerToolExecutionContext(execute, () => definition.execute === execute);
		const factoryTool: AgentTool<typeof parameters> = wrapToolDefinition(definition);
		const settingsManager = SettingsManager.create(tempDir, agentDir);
		const resourceLoader = new DefaultResourceLoader({
			cwd: tempDir,
			agentDir,
			settingsManager,
			extensionFactories: [(pi) => pi.registerTool(factoryTool)],
		});
		await resourceLoader.reload();
		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir,
			model: getModel("anthropic", "claude-sonnet-4-5")!,
			settingsManager,
			sessionManager: SessionManager.inMemory(),
			resourceLoader,
		});
		try {
			const registered = session.agent.state.tools.find((tool) => tool.name === "native_factory")!;
			await registered.execute("legacy", {});
			const context = createToolExecutionContext(registered.execute, {
				isValid: () => true,
				evaluate: () => "allow",
				approve: async () => false,
			}, () => session.agent.state.tools.includes(registered));
			await registered.execute("native", {}, undefined, undefined, context);
			expect(received).toBe(context);
			expect(effects).toBe(1);
			context.finish();
			context.close();
		} finally {
			session.dispose();
		}
	});

	it("refreshes tool registry when tools are registered after initialization", async () => {
		const settingsManager = SettingsManager.create(tempDir, agentDir);
		const sessionManager = SessionManager.inMemory();

		const resourceLoader = new DefaultResourceLoader({
			cwd: tempDir,
			agentDir,
			settingsManager,
			extensionFactories: [
				(pi) => {
					pi.on("session_start", () => {
						pi.registerTool({
							name: "dynamic_tool",
							label: "Dynamic Tool",
							description: "Tool registered from session_start",
							promptSnippet: "Run dynamic test behavior",
							promptGuidelines: ["Use dynamic_tool when the user asks for dynamic behavior tests."],
							parameters: Type.Object({}),
							execute: async () => ({
								content: [{ type: "text", text: "ok" }],
								details: {},
							}),
						});
					});
				},
			],
		});
		await resourceLoader.reload();

		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir,
			model: getModel("anthropic", "claude-sonnet-4-5")!,
			settingsManager,
			sessionManager,
			resourceLoader,
		});

		expect(session.getAllTools().map((tool) => tool.name)).not.toContain("dynamic_tool");

		await session.bindExtensions({});

		const allTools = session.getAllTools();
		const dynamicTool = allTools.find((tool) => tool.name === "dynamic_tool");
		const readTool = allTools.find((tool) => tool.name === "read");

		expect(allTools.map((tool) => tool.name)).toContain("dynamic_tool");
		expect(dynamicTool?.promptGuidelines).toEqual([
			"Use dynamic_tool when the user asks for dynamic behavior tests.",
		]);
		expect(dynamicTool?.sourceInfo).toMatchObject({
			path: "<inline:1>",
			source: "inline",
			scope: "temporary",
			origin: "top-level",
		});
		expect(readTool?.sourceInfo).toMatchObject({
			path: "builtin:read",
			source: "builtin",
			scope: "temporary",
			origin: "top-level",
		});
		expect(session.getActiveToolNames()).toContain("dynamic_tool");
		expect(session.systemPrompt).toContain("- dynamic_tool: Run dynamic test behavior");
		expect(session.systemPrompt).toContain("- Use dynamic_tool when the user asks for dynamic behavior tests.");

		session.dispose();
	});

	it("returns source metadata for SDK custom tools", async () => {
		const settingsManager = SettingsManager.create(tempDir, agentDir);
		const sessionManager = SessionManager.inMemory();
		const resourceLoader = new DefaultResourceLoader({
			cwd: tempDir,
			agentDir,
			settingsManager,
		});
		await resourceLoader.reload();

		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir,
			model: getModel("anthropic", "claude-sonnet-4-5")!,
			settingsManager,
			sessionManager,
			resourceLoader,
			customTools: [
				{
					name: "sdk_tool",
					label: "SDK Tool",
					description: "Tool registered through createAgentSession",
					parameters: Type.Object({}),
					execute: async () => ({
						content: [{ type: "text", text: "ok" }],
						details: {},
					}),
				},
			],
		});

		const sdkTool = session.getAllTools().find((tool) => tool.name === "sdk_tool");
		expect(sdkTool?.sourceInfo).toMatchObject({
			path: "<sdk:sdk_tool>",
			source: "sdk",
			scope: "temporary",
			origin: "top-level",
		});
		expect(session.getActiveToolNames()).toContain("sdk_tool");

		session.dispose();
	});

	it("keeps custom tools active but omits them from available tools when promptSnippet is not provided", async () => {
		const settingsManager = SettingsManager.create(tempDir, agentDir);
		const sessionManager = SessionManager.inMemory();

		const resourceLoader = new DefaultResourceLoader({
			cwd: tempDir,
			agentDir,
			settingsManager,
			extensionFactories: [
				(pi) => {
					pi.on("session_start", () => {
						pi.registerTool({
							name: "hidden_tool",
							label: "Hidden Tool",
							description: "Description should not appear in available tools",
							parameters: Type.Object({}),
							execute: async () => ({
								content: [{ type: "text", text: "ok" }],
								details: {},
							}),
						});
					});
				},
			],
		});
		await resourceLoader.reload();

		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir,
			model: getModel("anthropic", "claude-sonnet-4-5")!,
			settingsManager,
			sessionManager,
			resourceLoader,
		});

		await session.bindExtensions({});

		expect(session.getAllTools().map((tool) => tool.name)).toContain("hidden_tool");
		expect(session.getActiveToolNames()).toContain("hidden_tool");
		expect(session.systemPrompt).not.toContain("hidden_tool");
		expect(session.systemPrompt).not.toContain("Description should not appear in available tools");

		session.dispose();
	});
});
