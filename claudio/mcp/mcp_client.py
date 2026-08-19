from langchain_mcp_adapters.client import MultiServerMCPClient
from claudio.mcp.mcp_config import load_mcp_configs
from claudio.observability.logging import get_logger

logger = get_logger(__name__)

async def get_mcp_tools() -> list:
  """Connect to all configured MCP servers and return their tools.

  Failures are non-fatal: if the config is missing or a server can't be reached
  (common on a fresh machine without the MCP server binaries), we log and return
  no tools so the agent still runs with its built-in tools."""
  try:
      configs = load_mcp_configs()
  except FileNotFoundError:
      logger.info("No MCP config found — skipping MCP tools")
      return []
  if not configs:
      logger.info("No MCP servers configured — skipping MCP tools")
      return []

  logger.info(f"Connecting to MCP servers: {list(configs.keys())}")
  try:
      client = MultiServerMCPClient(configs)
      tools = await client.get_tools()
      logger.info(f"Loaded {len(tools)} tools from MCP servers")
      return tools
  except Exception as e:  # noqa: BLE001 — MCP is optional; never block startup
      logger.warning(f"Could not load MCP tools ({e}) — continuing without them")
      return []