local dap = require("dap")

dap.adapters.codelldb = {
	type = "server",
	port = "${port}",
	executable = {
		command = vim.fn.stdpath("data") .. "/mason/bin/codelldb",
		args = { "--port", "${port}" },
	},
}

dap.configurations.rust = {
	{
		name = "extension-debug",
		type = "codelldb",
		request = "launch",
		sourceLanguages = { "rust", "c++" },
		program = function()
			return vim.fn.getcwd() .. "/duckdb-rs/duckdb/build/debug/duckdb"
		end,
		args = { "-unsigned", "-cmd \"set allow_extensions_metadata_mismatch=true;load 'quack_ml.duckdb_extension'\"" },
		stopOnEntry = false,
		env = {
			LD_LIBRARY_PATH = vim.fn.getcwd(),
		},
	},
}
