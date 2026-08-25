from typing import Optional
from pydantic import BaseModel, Field


class QueryClassification(BaseModel):
    category: str = Field(
        description="One of: LOW_CHUNKS, HIGH_CHUNKS, GRAPH_RETRIEVAL"
    )
    reason: str = Field(description="Brief explanation of why this category was chosen")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )


class ImpactQuery(BaseModel):
    entity_name: str = Field(description="The policy entity being changed")
    entity_type: str = Field(
        description="Node label: Policy, Requirement, Entity, Role, Process, Consequence"
    )
    change_type: str = Field(
        description="Type of change: modify, remove, add, increase, decrease"
    )
    current_value: Optional[str] = Field(
        default=None, description="Current value if mentioned"
    )
    proposed_value: Optional[str] = Field(
        default=None, description="Proposed new value if mentioned"
    )


class ImpactItem(BaseModel):
    affected_entity: str = Field(description="Name of the affected entity")
    entity_type: str = Field(description="Type/label of the affected entity")
    impact_level: str = Field(description="'direct' or 'indirect'")
    relationship_path: str = Field(
        description="Human-readable explanation of the dependency chain"
    )
    severity: str = Field(description="'high', 'medium', or 'low'")


class ImpactAnalysisResult(BaseModel):
    analyzed_entity: str = Field(description="The entity that was analyzed")
    change_description: str = Field(description="Description of the proposed change")
    entities_found: bool = Field(
        description="Whether the entity was found in the graph"
    )
    direct_impacts: list[ImpactItem] = Field(default_factory=list)
    indirect_impacts: list[ImpactItem] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name")
    label: str = Field(
        description="Node label: Policy, Requirement, Entity, Role, Process, Consequence"
    )
    properties: dict = Field(
        default_factory=dict, description="Additional properties for this entity"
    )


class ExtractedRelationship(BaseModel):
    source_name: str = Field(description="Source entity name")
    source_label: str = Field(description="Source entity label")
    relationship_type: str = Field(
        description="Relationship type: DEFINES, DEPENDS_ON, APPLIES_TO, TRIGGERS, RESULTS_IN, REFERENCES, AFFECTS, GOVERNS, HAS_EXCEPTION"
    )
    target_name: str = Field(description="Target entity name")
    target_label: str = Field(description="Target entity label")
    properties: dict = Field(
        default_factory=dict, description="Additional relationship properties"
    )


class GraphExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
