/**
 * Tool wrappers for extension-registered tools.
 *
 * These wrappers only adapt tool execution so extension tools receive the runner context.
 * Tool call and tool result interception is handled by AgentSession via agent-core hooks.
 */

import { type AgentTool, bindToolExecutionContext, type ToolExecutionContext } from "@earendil-works/pi-agent-core";
import { wrapToolDefinition } from "../tools/tool-definition-wrapper.ts";
import type { ExtensionRunner } from "./runner.ts";
import type { RegisteredTool } from "./types.ts";

/**
 * Wrap a RegisteredTool into an AgentTool.
 * Uses the runner's createToolContext() for consistent context across tools and event handlers.
 */
export function wrapRegisteredTool(
	registeredTool: RegisteredTool,
	runner: Pick<ExtensionRunner, "createToolContext">,
): AgentTool {
	const definition = registeredTool.definition;
	const tool = wrapToolDefinition(definition, (toolCallId, signal) => runner.createToolContext(toolCallId, signal));
	const delegate = tool.execute;
	const execute: AgentTool["execute"] = async (toolCallId, params, signal, onUpdate, context?: ToolExecutionContext) => {
		context?.assertValid(execute);
		return context
			? delegate.call(tool, toolCallId, params, signal, onUpdate, context)
			: delegate.call(tool, toolCallId, params, signal, onUpdate);
	};
	bindToolExecutionContext(execute, delegate, () => registeredTool.definition === definition && tool.execute === delegate);
	return { ...tool, execute };
}

/**
 * Wrap all registered tools into AgentTools.
 * Uses the runner's createToolContext() for consistent context across tools and event handlers.
 */
export function wrapRegisteredTools(registeredTools: RegisteredTool[], runner: ExtensionRunner): AgentTool[] {
	return registeredTools.map((tool) => wrapRegisteredTool(tool, runner));
}
