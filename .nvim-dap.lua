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
			vim.fn.setenv("OMP_NUM_THREADS", "1")
			return vim.fn.getcwd() .. "/duckdb-rs/duckdb/build/debug/duckdb"
		end,
		args = { "-unsigned", "-cmd", "set allow_extensions_metadata_mismatch=true;load 'quack_ml.duckdb_extension'" },
		stopOnEntry = false,
		postRunCommands = {
			"target modules add quack_ml.duckdb_extension",
		},
	},
}

dap.configurations.rust[1] = setmetatable(dap.configurations.rust[1], {
	__call = function(config)
		vim.fn.system("cargo build")
		vim.fn.system("cp ./target/debug/libquack_ml.dylib quack_ml.duckdb_extension")
		return config
	end,
})
