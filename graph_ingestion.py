import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
from graph_schema import GraphExtractionResult, ExtractedEntity, ExtractedRelationship

load_dotenv()

EXTRACTION_PROMPT = """You are a knowledge graph extraction engine for academic/organizational policy documents.

From the text below, extract:
1. ENTITIES - Named things that can be nodes in a graph:
   - Policy: A named policy, regulation, or rule set (e.g., "Attendance Policy", "Grading Policy")
   - Requirement: A specific rule with a threshold or condition (e.g., "Minimum Attendance 70%", "Minimum GPA 2.0")
   - Entity: A domain concept referenced by policies (e.g., "GPA", "Credit Hours", "Semester")
   - Role: A stakeholder or actor (e.g., "Student", "Faculty", "Dean", "Department Head")
   - Process: A procedure or workflow (e.g., "Academic Probation", "Grade Appeal", "Attendance Verification")
   - Consequence: An outcome when requirements are met or violated (e.g., "Examination Eligibility", "Dismissal", "Scholarship Loss")

2. RELATIONSHIPS between entities:
   - DEFINES: Policy defines a Requirement
   - DEPENDS_ON: One Requirement depends on another
   - APPLIES_TO: Requirement applies to a Role
   - TRIGGERS: Requirement triggers a Process (when met or violated)
   - RESULTS_IN: Process results in a Consequence
   - REFERENCES: Policy references another Policy
   - AFFECTS: One Entity affects another Entity
   - GOVERNS: Policy governs an Entity
   - HAS_EXCEPTION: Requirement has an exception (another Requirement)

Rules:
- Use clear, normalized names (e.g., "Minimum Attendance Requirement" not "the attendance rule mentioned above")
- Include threshold values in Requirement properties when present (e.g., threshold_value: "70%", threshold_type: "minimum")
- Only extract relationships that are clearly stated or strongly implied by the text
- Do not invent relationships that are not supported by the content

TEXT TO ANALYZE:
{chunk_text}"""


def _extract_from_chunk(chunk_text: str, llm) -> GraphExtractionResult:
    structured_llm = llm.with_structured_output(GraphExtractionResult)
    messages = [
        SystemMessage(content=EXTRACTION_PROMPT.format(chunk_text=chunk_text)),
        HumanMessage(content="Extract all entities and relationships from this text."),
    ]
    return structured_llm.invoke(messages)


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _resolve_entities(
    all_entities: list[ExtractedEntity],
) -> dict[str, ExtractedEntity]:
    resolved = {}
    for entity in all_entities:
        key = _normalize_name(entity.name)
        if key in resolved:
            existing = resolved[key]
            for prop_key, prop_val in entity.properties.items():
                if prop_key not in existing.properties or not existing.properties[prop_key]:
                    existing.properties[prop_key] = prop_val
        else:
            resolved[key] = entity
    return resolved


def _populate_neo4j(
    entities: dict[str, ExtractedEntity],
    relationships: list[ExtractedRelationship],
):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session(database=NEO4J_DATABASE) as session:
        for key, entity in entities.items():
            props = {"name": entity.name, **entity.properties}
            props_str = ", ".join(
                f"n.{k} = ${k}" for k in props if k != "name"
            )
            query = f"MERGE (n:{entity.label} {{name: $name}})"
            if props_str:
                query += f" SET {props_str}"
            session.run(query, **props)

        for rel in relationships:
            source_key = _normalize_name(rel.source_name)
            target_key = _normalize_name(rel.target_name)
            if source_key not in entities or target_key not in entities:
                continue

            source_label = entities[source_key].label
            target_label = entities[target_key].label

            props_str = ""
            if rel.properties:
                props_str = " {" + ", ".join(
                    f"{k}: ${k}" for k in rel.properties
                ) + "}"

            query = (
                f"MATCH (a:{source_label} {{name: $source_name}}), "
                f"(b:{target_label} {{name: $target_name}}) "
                f"MERGE (a)-[r:{rel.relationship_type}{props_str}]->(b)"
            )
            params = {
                "source_name": rel.source_name,
                "target_name": rel.target_name,
                **rel.properties,
            }
            session.run(query, **params)

    driver.close()


def run_graph_ingestion(documents: list, batch_size: int = 5):
    print("Starting knowledge graph ingestion...")

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
        temperature=0,
        max_tokens=4096,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    all_entities: list[ExtractedEntity] = []
    all_relationships: list[ExtractedRelationship] = []
    failed_chunks = 0

    for i, doc in enumerate(documents):
        print(f"  Extracting graph data from chunk {i + 1}/{len(documents)}...")
        try:
            result = _extract_from_chunk(doc.page_content, llm)
            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)
        except Exception as e:
            print(f"    Extraction failed for chunk {i + 1}: {e}")
            failed_chunks += 1
            continue

    if not all_entities:
        print("No entities extracted. Skipping graph population.")
        return

    print(f"  Extracted {len(all_entities)} entities, {len(all_relationships)} relationships.")
    if failed_chunks:
        print(f"  Skipped {failed_chunks} chunk(s) due to extraction errors.")

    resolved_entities = _resolve_entities(all_entities)
    print(f"  Resolved to {len(resolved_entities)} unique entities after deduplication.")

    print("  Populating Neo4j...")
    _populate_neo4j(resolved_entities, all_relationships)
    print("Knowledge graph ingestion complete.")
