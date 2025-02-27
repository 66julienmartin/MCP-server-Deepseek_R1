# Deepseek R1 MCP Server
[![smithery badge](https://smithery.ai/badge/@66julienmartin/mcp-server-deepseek_r1)](https://smithery.ai/server/@66julienmartin/mcp-server-deepseek_r1)

A Model Context Protocol (MCP) server implementation for the Deepseek R1 language model. Deepseek R1 is a powerful language model optimized for reasoning tasks with a context window of 8192 tokens.

Why Node.js?
This implementation uses Node.js/TypeScript as it provides the most stable integration with MCP servers. The Node.js SDK offers better type safety, error handling, and compatibility with Claude Desktop.

<a href="https://glama.ai/mcp/servers/qui5thpyvu"><img width="380" height="200" src="https://glama.ai/mcp/servers/qui5thpyvu/badge" alt="Deepseek R1 Server MCP server" /></a>

## Quick Start

### Installing via Smithery

To install Deepseek R1 for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@66julienmartin/mcp-server-deepseek_r1):

```bash
npx -y @smithery/cli install @66julienmartin/mcp-server-deepseek_r1 --client claude
```

### Installing manually
```bash
# Clone and install
git clone https://github.com/66julienmartin/MCP-server-Deepseek_R1.git
cd deepseek-r1-mcp
npm install

# Set up environment
cp .env.example .env  # Then add your API key

# Build and run
npm run build
```

## Prerequisites

- Node.js (v18 or higher)
- npm
- Claude Desktop
- Deepseek API key

## Model Selection

By default, this server uses the **deepseek-R1** model. If you want to use **DeepSeek-V3** instead, modify the model name in `src/index.ts`:

```typescript
// For DeepSeek-R1 (default)
model: "deepseek-reasoner"

// For DeepSeek-V3
model: "deepseek-chat"
```

## Project Structure

```
deepseek-r1-mcp/
├── src/
│   ├── index.ts             # Main server implementation
├── build/                   # Compiled files
│   ├── index.js
├── LICENSE
├── README.md
├── package.json
├── package-lock.json
└── tsconfig.json
```

## Configuration

1. Create a `.env` file:
```
DEEPSEEK_API_KEY=your-api-key-here
```

2. Update Claude Desktop configuration:
```json
{
  "mcpServers": {
    "deepseek_r1": {
      "command": "node",
      "args": ["/path/to/deepseek-r1-mcp/build/index.js"],
      "env": {
        "DEEPSEEK_API_KEY": "your-api-key"
      }
    }
  }
}
```

## Development

```bash
npm run dev     # Watch mode
npm run build   # Build for production
```

## Features

- Advanced text generation with Deepseek R1 (8192 token context window)
- Configurable parameters (max_tokens, temperature)
- Robust error handling with detailed error messages
- Full MCP protocol support
- Claude Desktop integration
- Support for both DeepSeek-R1 and DeepSeek-V3 models

## API Usage

```typescript
{
  "name": "deepseek_r1",
  "arguments": {
    "prompt": "Your prompt here",
    "max_tokens": 8192,    // Maximum tokens to generate
    "temperature": 0.2     // Controls randomness
  }
}
```

## The Temperature Parameter

The default value of `temperature` is 0.2.

Deepseek recommends setting the `temperature` according to your specific use case:

| USE CASE | TEMPERATURE | EXAMPLE |
|----------|-------------|---------|
| Coding / Math | 0.0 | Code generation, mathematical calculations |
| Data Cleaning / Data Analysis | 1.0 | Data processing tasks |
| General Conversation | 1.3 | Chat and dialogue |
| Translation | 1.3 | Language translation |
| Creative Writing / Poetry | 1.5 | Story writing, poetry generation |

## Error Handling

The server provides detailed error messages for common issues:
- API authentication errors
- Invalid parameters
- Rate limiting
- Network issues

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
