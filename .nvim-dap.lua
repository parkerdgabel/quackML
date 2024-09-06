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
		args = { "-unsigned", "-cmd", "set allow_extensions_metadata_mismatch=true;load 'quack_ml.duckdb_extension'" },
		stopOnEntry = false,
		postRunCommands = {
			"target modules add quack_ml.duckdb_extension",
		},
	},
}

dap.configurations.rust[1] = setmetatable(dap.configurations.rust[1], {
	__call = function(config)
		vim.fn.system(
			'RUSTFLAGS="-C link-arg=-Wl,-rpath,/usr/lib -L /opt/homebrew/opt/libomp/lib -L /opt/homebrew/Cellar/python@3.11/3.11.9_1/Frameworks/Python.framework/Versions/3.11/lib" cargo build'
		)
		vim.fn.system("cp ./target/debug/libquack_ml.dylib quack_ml.duckdb_extension")
		return config
	end,
})
