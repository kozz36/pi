import {
	type AgentTool,
	type AgentToolUpdateCallback,
	bindToolExecutionContext,
	isToolExecutionContext,
	type ToolExecutionContext,
} from "@earendil-works/pi-agent-core";
import type { TSchema } from "typebox";
import type { ExtensionToolContext, ToolDefinition } from "../extensions/types.ts";

/** Creates the context for one tool call. */
export type ToolContextFactory = (toolCallId: string, signal: AbortSignal | undefined) => ExtensionToolContext;

/** Wrap a ToolDefinition into an AgentTool for the core runtime. */
export function wrapToolDefinition<TDetails = unknown, TParams extends TSchema = TSchema>(
	definition: ToolDefinition<TParams, TDetails>,
	ctxFactory?: ToolContextFactory,
) {
	const delegate = definition.execute;
	const execute = async (
		toolCallId: string,
		params: Parameters<typeof delegate>[1],
		signal?: AbortSignal,
		onUpdate?: AgentToolUpdateCallback<TDetails>,
		ctx?: ExtensionToolContext | ToolExecutionContext,
		context?: ToolExecutionContext,
	) => {
		// A factory result can itself be registered as a definition: extension
		// context then remains fifth and native context arrives separately sixth.
		if (context !== undefined && !isToolExecutionContext(context)) {
			throw new Error("Expected an SDK tool execution context");
		}
		if (isToolExecutionContext(ctx) && context !== undefined && ctx !== context) {
			throw new Error("Conflicting SDK tool execution contexts");
		}
		const nativeContext = context ?? (isToolExecutionContext(ctx) ? ctx : undefined);
		nativeContext?.assertValid(execute);
		const extensionContext = isToolExecutionContext(ctx) ? undefined : ctx;
		const resolvedContext = extensionContext ?? ctxFactory?.(toolCallId, signal);
		// The legacy definition ABI always had five arguments, including absent UI context.
		if (!nativeContext) {
			return definition.execute(toolCallId, params, signal, onUpdate, resolvedContext as ExtensionToolContext);
		}
		return delegate.call(
			definition, toolCallId, params, signal, onUpdate, resolvedContext as ExtensionToolContext, nativeContext,
		);
	};
	bindToolExecutionContext(execute, delegate, () => definition.execute === delegate);
	return {
		name: definition.name,
		label: definition.label,
		description: definition.description,
		parameters: definition.parameters,
		outputSchema: definition.outputSchema,
		constrainedSampling: definition.constrainedSampling,
		prepareArguments: definition.prepareArguments,
		executionMode: definition.executionMode,
		execute,
	} satisfies AgentTool<TParams, TDetails>;
}

/** Wrap multiple ToolDefinitions into AgentTools for the core runtime. */
export function wrapToolDefinitions(
	definitions: ToolDefinition<any, any>[],
	ctxFactory?: ToolContextFactory,
): AgentTool<any>[] {
	return definitions.map((definition) => wrapToolDefinition(definition, ctxFactory));
}

/**
 * Synthesize a minimal ToolDefinition from an AgentTool.
 *
 * This keeps AgentSession's internal registry definition-first even when a caller
 * provides plain AgentTool overrides that do not include prompt metadata or renderers.
 */
export function createToolDefinitionFromAgentTool<TParams extends TSchema>(
	tool: AgentTool<TParams>,
): ToolDefinition<TParams, unknown> {
	const delegate = tool.execute;
	const execute: ToolDefinition<TParams, unknown>["execute"] = async (
		toolCallId,
		params,
		signal,
		onUpdate,
		_ctx,
		context,
	) => {
		if (context !== undefined) {
			if (!isToolExecutionContext(context)) throw new Error("Expected an SDK tool execution context");
			context.assertValid(execute);
			return delegate.call(tool, toolCallId, params, signal, onUpdate, context);
		}
		return tool.execute(toolCallId, params, signal, onUpdate);
	};
	bindToolExecutionContext(execute, delegate, () => tool.execute === delegate);
	return {
		name: tool.name,
		label: tool.label,
		description: tool.description,
		parameters: tool.parameters,
		outputSchema: tool.outputSchema,
		constrainedSampling: tool.constrainedSampling,
		prepareArguments: tool.prepareArguments,
		executionMode: tool.executionMode,
		execute,
	};
}
