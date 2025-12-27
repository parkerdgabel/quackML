use anyhow::{Context, Result};

pub struct DatabaseContext {
    connection: duckdb::Connection,
}

impl DatabaseContext {
    // Method to create a new DatabaseContext with a cloned connection
    pub fn new(connection: &duckdb::Connection) -> Result<Self> {
        Ok(DatabaseContext {
            connection: connection
                .try_clone()
                .context("Failed to clone connection")?,
        })
    }

    // Method to get the connection (if needed)
    pub fn get_connection(&self) -> &duckdb::Connection {
        &self.connection
    }
}

pub static mut DATABASE_CONTEXT: Option<DatabaseContext> = None;

pub fn init_database_context(connection: &duckdb::Connection) -> Result<()> {
    unsafe {
        DATABASE_CONTEXT = Some(DatabaseContext::new(connection)?);
    }
    Ok(())
}

pub fn run<T, F>(f: F) -> Result<T>
where
    F: FnOnce(&duckdb::Connection) -> Result<T>,
{
    let database_context = unsafe {
        DATABASE_CONTEXT
            .as_ref()
            .context("Database context not initialized")?
    };
    f(database_context.get_connection())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_context_initialization() -> Result<()> {
        let connection =
            duckdb::Connection::open_in_memory().context("Failed to open in-memory connection")?;
        init_database_context(&connection)?;
        let database_context = unsafe { DATABASE_CONTEXT.as_ref().unwrap() };
        // Check if the connection path is the same
        assert_eq!(
            &database_context.get_connection().path(),
            &connection.path()
        );
        Ok(())
    }

    #[test]
    fn test_database_context_new() -> Result<()> {
        let connection =
            duckdb::Connection::open_in_memory().context("Failed to open in-memory connection")?;
        let context = DatabaseContext::new(&connection)?;
        // Verify we can get the connection back
        assert!(context.get_connection().path().is_none()); // In-memory has no path
        Ok(())
    }

    #[test]
    fn test_run_with_initialized_context() -> Result<()> {
        let connection =
            duckdb::Connection::open_in_memory().context("Failed to open in-memory connection")?;
        init_database_context(&connection)?;

        // Run a simple query using the context
        let result = run(|conn| {
            let value: i32 = conn.query_row("SELECT 1 + 1", [], |row| row.get(0))?;
            Ok(value)
        })?;

        assert_eq!(result, 2);
        Ok(())
    }

    #[test]
    fn test_run_returns_closure_result() -> Result<()> {
        let connection =
            duckdb::Connection::open_in_memory().context("Failed to open in-memory connection")?;
        init_database_context(&connection)?;

        // Test that run properly returns the closure's result
        let result = run(|_conn| Ok("test string"))?;
        assert_eq!(result, "test string");

        let result = run(|_conn| Ok(42))?;
        assert_eq!(result, 42);

        Ok(())
    }

    #[test]
    fn test_run_propagates_errors() -> Result<()> {
        let connection =
            duckdb::Connection::open_in_memory().context("Failed to open in-memory connection")?;
        init_database_context(&connection)?;

        // Test that errors from the closure are propagated
        let result: Result<i32> = run(|_conn| {
            anyhow::bail!("intentional error")
        });

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("intentional error"));
        Ok(())
    }

    #[test]
    fn test_database_context_execute_query() -> Result<()> {
        let connection =
            duckdb::Connection::open_in_memory().context("Failed to open in-memory connection")?;
        init_database_context(&connection)?;

        // Create a table and insert data using the context
        run(|conn| {
            conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)", [])?;
            conn.execute("INSERT INTO test_table VALUES (1, 'Alice'), (2, 'Bob')", [])?;
            Ok(())
        })?;

        // Query the data
        let count: i32 = run(|conn| {
            conn.query_row("SELECT COUNT(*) FROM test_table", [], |row| row.get(0))
                .map_err(|e| anyhow::anyhow!(e))
        })?;

        assert_eq!(count, 2);
        Ok(())
    }

    #[test]
    fn test_get_connection_returns_valid_connection() -> Result<()> {
        let connection =
            duckdb::Connection::open_in_memory().context("Failed to open in-memory connection")?;
        let context = DatabaseContext::new(&connection)?;

        // Verify the connection works by executing a query
        let result: i32 = context
            .get_connection()
            .query_row("SELECT 42", [], |row| row.get(0))?;

        assert_eq!(result, 42);
        Ok(())
    }
}
