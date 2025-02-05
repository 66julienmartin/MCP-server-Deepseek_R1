#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { ListToolsRequestSchema, CallToolRequestSchema, ErrorCode, McpError } from "@modelcontextprotocol/sdk/types.js";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const API_KEY = process.env.DEEPSEEK_API_KEY;
if (!API_KEY) {
    throw new Error("DEEPSEEK_API_KEY environment variable is required");
}

interface ChatCompletionArgs {
    prompt: string;
    max_tokens?: number;
    temperature?: number;
}

function isValidChatCompletionArgs(args: unknown): args is ChatCompletionArgs {
    return (
        typeof args === "object" &&
        args !== null &&
        "prompt" in args &&
        typeof (args as ChatCompletionArgs).prompt === "string" &&
        ((args as ChatCompletionArgs).max_tokens === undefined || typeof (args as ChatCompletionArgs).max_tokens === "number") &&
        ((args as ChatCompletionArgs).temperature === undefined || typeof (args as ChatCompletionArgs).temperature === "number")
    );
}

class DeepseekR1Server {
    private server: Server;
    private openai: OpenAI;

    constructor() {
        this.server = new Server(
            { name: "deepseek_r1", version: "1.0.0" },
            { capabilities: { tools: {} } }
        );

        this.openai = new OpenAI({
            apiKey: API_KEY,
            baseURL: "https://api.deepseek.com"
        });

        this.setupHandlers();
        this.setupErrorHandling();
    }

    private setupErrorHandling(): void {
        this.server.onerror = (error: Error) => {
            console.error("[MCP Error]", error);
        };

        process.on("SIGINT", async () => {
            await this.server.close();
            process.exit(0);
        });
    }

    private setupHandlers(): void {
        this.server.setRequestHandler(
            ListToolsRequestSchema,
            async () => ({
                tools: [{
                    name: "deepseek_r1",
                    description: "Generate text using DeepSeek R1 model",
                    inputSchema: {
                        type: "object",
                        properties: {
                            prompt: {
                                type: "string",
                                description: "Input text for DeepSeek"
                            },
                            max_tokens: {
                                type: "number",
                                description: "Maximum tokens to generate (default: 8192)",
                                minimum: 1,
                                maximum: 8192
                            },
                            temperature: {
                                type: "number",
                                description: "Sampling temperature (default: 0.2)",
                                minimum: 0,
                                maximum: 2
                            }
                        },
                        required: ["prompt"]
                    }
                }]
            })
        );

        this.server.setRequestHandler(
            CallToolRequestSchema,
            async (request) => {
                if (request.params.name !== "deepseek_r1") {
                    throw new McpError(
                        ErrorCode.MethodNotFound,
                        `Unknown tool: ${request.params.name}`
                    );
                }

                if (!isValidChatCompletionArgs(request.params.arguments)) {
                    throw new McpError(
                        ErrorCode.InvalidParams,
                        "Invalid chat completion arguments"
                    );
                }

                try {
                    const completion = await this.openai.chat.completions.create({
                        model: "deepseek-reasoner",
                        messages: [
                            {
                                role: "system",
                                content: "Vous êtes un assistant intelligent et polyvalent."
                            },
                            {
                                role: "user",
                                content: request.params.arguments.prompt
                            }
                        ],
                        max_tokens: request.params.arguments.max_tokens ?? 8192,
                        temperature: request.params.arguments.temperature ?? 0.2
                    });

                    return {
                        content: [{
                            type: "text",
                            text: completion.choices[0]?.message?.content || "No response"
                        }]
                    };
                } catch (error) {
                    console.error("DeepSeek API error:", error);
                    return {
                        content: [{
                            type: "text",
                            text: `DeepSeek API error: ${error instanceof Error ? error.message : String(error)}`
                        }],
                        isError: true
                    };
                }
            }
        );
    }

    async run(): Promise<void> {
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
        console.error("DeepSeek R1 MCP server running on stdio");
    }
}

const server = new DeepseekR1Server();
server.run().catch(console.error);