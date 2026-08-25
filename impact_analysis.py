import json
import re
from langchain_core.messages import HumanMessage
from graph_schema import ImpactAnalysisResult


def _format_impacts(impacts: list) -> str:
    if not impacts:
        return "None identified."
    lines = []
    for i, item in enumerate(impacts, 1):
        lines.append(
            f"{i}. {item.affected_entity} ({item.entity_type}) "
            f"[Severity: {item.severity}]\n"
            f"   Path: {item.relationship_path}"
        )
    return "\n".join(lines)


IMPACT_ANALYSIS_PROMPT = """You are a policy impact analyst. Based on the knowledge graph analysis below, provide a structured impact assessment.

IMPORTANT OUTPUT RULES:
- Return only the final analysis for the user.
- Do not show or mention your reasoning, analysis, chain of thought, or thinking process.
- Never output <think> tags or any content inside them.

PROPOSED CHANGE:
{change_description}

DIRECT IMPACTS (immediately affected policies/entities):
{direct_impacts}

INDIRECT/CASCADING IMPACTS (downstream effects through dependency chains):
{indirect_impacts}

SUPPORTING DOCUMENT EVIDENCE:
{evidence}

Provide your analysis in this structure:

**Change:** {change_description}

**Direct Impacts:**
(List each directly affected entity with a brief explanation of why it is affected)

**Indirect/Cascading Impacts:**
(List each indirectly affected entity with the dependency chain explanation)

**Risk Assessment:**
(Overall assessment of the scope and severity of this change)

**Recommendations:**
(Any policies or entities that should be reviewed or updated)

If insufficient graph data exists for any section, clearly state what could not be determined. Do not invent relationships that are not present in the data above."""


def generate_impact_answer(
    impact_result: ImpactAnalysisResult,
    supporting_chunks: list,
    query: str,
    llm,
    chat_history=None,
) -> str:
    try:
        evidence_text = ""
        if supporting_chunks:
            for i, chunk in enumerate(supporting_chunks):
                if "original_content" in chunk.metadata:
                    original_data = json.loads(chunk.metadata["original_content"])
                    raw_text = original_data.get("raw_text", "")
                    if raw_text:
                        evidence_text += f"--- Document {i + 1} ---\n{raw_text}\n\n"

        if not evidence_text:
            evidence_text = "No supporting document evidence retrieved."

        prompt_text = IMPACT_ANALYSIS_PROMPT.format(
            change_description=impact_result.change_description,
            direct_impacts=_format_impacts(impact_result.direct_impacts),
            indirect_impacts=_format_impacts(impact_result.indirect_impacts),
            evidence=evidence_text,
        )

        if chat_history:
            prompt_text = (
                "Conversation history:\n"
                + "\n".join(
                    f"{message.type.capitalize()}: {message.content}"
                    for message in chat_history[-4:]
                )
                + "\n\n"
                + prompt_text
            )

        print("Generating impact analysis...")
        message = HumanMessage(content=prompt_text)
        response = llm.invoke([message])

        text = response.content
        if isinstance(text, list):
            text = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in text
            )
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<think>.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    except Exception as e:
        print(f"Impact analysis generation failed: {e}")
        return (
            f"Impact analysis for: {impact_result.change_description}\n\n"
            f"Direct impacts: {len(impact_result.direct_impacts)} identified\n"
            f"Indirect impacts: {len(impact_result.indirect_impacts)} identified\n\n"
            "Unable to generate detailed analysis. Please try again."
        )
