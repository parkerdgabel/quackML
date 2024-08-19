
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

pub fn init_database_context(connection: duckdb::Connection) {
    unsafe {
        DATABASE_CONTEXT = Some(DatabaseContext::new(&connection));
    }
}