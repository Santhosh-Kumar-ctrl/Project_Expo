// Neo4j schema setup for Policy Knowledge Graph
// Run this once to create constraints and indexes.
//
// Usage (Neo4j Browser or cypher-shell):
//   Copy and paste each statement individually, or run via:
//   cat neo4j_setup.cypher | cypher-shell -u neo4j -p <password>

// Uniqueness constraints (one per node label)
CREATE CONSTRAINT policy_name IF NOT EXISTS FOR (n:Policy) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT requirement_name IF NOT EXISTS FOR (n:Requirement) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT role_name IF NOT EXISTS FOR (n:Role) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT process_name IF NOT EXISTS FOR (n:Process) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT consequence_name IF NOT EXISTS FOR (n:Consequence) REQUIRE n.name IS UNIQUE;

// Full-text index for fuzzy entity search
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
FOR (n:Policy|Requirement|Entity|Role|Process|Consequence)
ON EACH [n.name, n.description];
