
pub struct DatabaseContext {
    connection: duckdb::Connection,
}

impl DatabaseContext {
    // Method to create a new DatabaseContext with a cloned connection
    pub fn new(connection: &duckdb::Connection) -> Self {
        DatabaseContext {
            connection: connection.try_clone().unwrap(),
        }
    }

    // Method to get the connection (if needed)
    pub fn get_connection(&self) -> &duckdb::Connection {
        &self.connection
    }
}

pub static mut DATABASE_CONTEXT: Option<DatabaseContext> = None;

pub fn init_database_context(connection: &duckdb::Connection) {
    unsafe {
        DATABASE_CONTEXT = Some(DatabaseContext::new(connection));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_context() {
        let connection = duckdb::Connection::open_in_memory().unwrap();
        init_database_context(&connection);
        let database_context = unsafe { DATABASE_CONTEXT.as_ref().unwrap() };
        // Check if the connection is the same
        assert_eq!(&database_context.get_connection().path(), &connection.path());
    }
}