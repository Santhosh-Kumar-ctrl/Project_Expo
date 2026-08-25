from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
from graph_schema import ImpactQuery, ImpactItem, ImpactAnalysisResult
from langchain_core.messages import HumanMessage, SystemMessage


ENTITY_EXTRACTION_PROMPT = """Extract the policy entity and proposed change from the query. Identify the entity name, its type (Policy/Requirement/Entity/Role/Process/Consequence), the change type (modify/remove/add/increase/decrease), and current/proposed values if mentioned."""


def _get_driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def _extract_impact_query(query: str, llm) -> ImpactQuery:
    from langchain_groq import ChatGroq
    import os

    extraction_llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        max_tokens=256,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    structured_llm = extraction_llm.with_structured_output(ImpactQuery)
    messages = [
        SystemMessage(content=ENTITY_EXTRACTION_PROMPT),
        HumanMessage(content=query),
    ]
    return structured_llm.invoke(messages)


def _find_entity_in_graph(tx, entity_name: str):
    result = tx.run(
        "MATCH (n) WHERE toLower(n.name) = toLower($name) RETURN n LIMIT 1",
        name=entity_name,
    )
    record = result.single()
    if record:
        return record["n"]

    result = tx.run(
        "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) RETURN n LIMIT 5",
        name=entity_name,
    )
    records = list(result)
    if records:
        return records[0]["n"]

    result = tx.run(
        "MATCH (n) WHERE toLower($name) CONTAINS toLower(n.name) RETURN n LIMIT 5",
        name=entity_name,
    )
    records = list(result)
    if records:
        return records[0]["n"]

    words = [w for w in entity_name.lower().split() if len(w) > 3]
    if words:
        result = tx.run(
            """MATCH (n)
            WITH n, size([word IN $words WHERE toLower(n.name) CONTAINS word]) as matches
            WHERE matches > 0
            RETURN n ORDER BY matches DESC LIMIT 5""",
            words=words,
        )
        records = list(result)
        if records:
            return records[0]["n"]

    return None


def _get_direct_impacts(tx, entity_name: str) -> list[dict]:
    result = tx.run(
        """MATCH (source {name: $name})-[r]-(target)
        WHERE source <> target
        RETURN source.name as source_name,
               labels(source)[0] as source_type,
               type(r) as relationship,
               target.name as target_name,
               labels(target)[0] as target_type,
               properties(r) as rel_props
        LIMIT 20""",
        name=entity_name,
    )
    return [dict(record) for record in result]


def _get_indirect_impacts(tx, entity_name: str) -> list[dict]:
    result = tx.run(
        """MATCH path = (source {name: $name})-[*2..3]-(target)
        WHERE source <> target
        RETURN DISTINCT target.name as target_name,
               labels(target)[0] as target_type,
               length(path) as depth,
               [r IN relationships(path) | type(r)] as path_relationships,
               [n IN nodes(path) | n.name] as path_nodes
        ORDER BY depth
        LIMIT 20""",
        name=entity_name,
    )
    return [dict(record) for record in result]


def perform_impact_analysis(query: str, llm) -> ImpactAnalysisResult:
    try:
        impact_query = _extract_impact_query(query, llm)
    except Exception as e:
        print(f"  Entity extraction failed: {e}")
        return ImpactAnalysisResult(
            analyzed_entity="unknown",
            change_description=query,
            entities_found=False,
        )

    change_desc = f"{impact_query.change_type} {impact_query.entity_name}"
    if impact_query.current_value and impact_query.proposed_value:
        change_desc += (
            f" from {impact_query.current_value} to {impact_query.proposed_value}"
        )
    elif impact_query.proposed_value:
        change_desc += f" to {impact_query.proposed_value}"

    try:
        driver = _get_driver()
        with driver.session(database=NEO4J_DATABASE) as session:
            entity_node = session.execute_read(
                _find_entity_in_graph, impact_query.entity_name
            )

            if entity_node is None:
                driver.close()
                return ImpactAnalysisResult(
                    analyzed_entity=impact_query.entity_name,
                    change_description=change_desc,
                    entities_found=False,
                )

            matched_name = entity_node["name"]
            print(f"  Matched graph entity: '{matched_name}'")

            direct_raw = session.execute_read(
                _get_direct_impacts, matched_name
            )
            indirect_raw = session.execute_read(
                _get_indirect_impacts, matched_name
            )

        driver.close()
    except Exception as e:
        print(f"  Neo4j query failed: {e}")
        return ImpactAnalysisResult(
            analyzed_entity=impact_query.entity_name,
            change_description=change_desc,
            entities_found=False,
        )

    direct_impacts = []
    for item in direct_raw:
        direct_impacts.append(
            ImpactItem(
                affected_entity=item["target_name"],
                entity_type=item["target_type"],
                impact_level="direct",
                relationship_path=f"{item['source_name']} --[{item['relationship']}]--> {item['target_name']}",
                severity="high",
            )
        )

    seen_indirect = set()
    indirect_impacts = []
    for item in indirect_raw:
        if item["target_name"] in seen_indirect:
            continue
        if any(d.affected_entity == item["target_name"] for d in direct_impacts):
            continue
        seen_indirect.add(item["target_name"])
        path_desc = " -> ".join(
            f"{node}" for node in item["path_nodes"] if node
        )
        indirect_impacts.append(
            ImpactItem(
                affected_entity=item["target_name"],
                entity_type=item["target_type"],
                impact_level="indirect",
                relationship_path=path_desc,
                severity="medium" if item["depth"] == 2 else "low",
            )
        )

    return ImpactAnalysisResult(
        analyzed_entity=impact_query.entity_name,
        change_description=change_desc,
        entities_found=True,
        direct_impacts=direct_impacts,
        indirect_impacts=indirect_impacts,
    )
