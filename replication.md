# RAKG Replication Notes

## Verified Functionality

### Document-Level KG Construction ✅
- **Status**: Working as described in the paper
- **Tested**: Processing individual documents from the MINE dataset (105 documents total)
- **Result**: Each document (1 out of 105) is successfully parsed into a knowledge graph
- **Output**: Separate KG files generated for each document (e.g., `1.json`, `2.json`, ..., `105.json`)

### Implementation Details

#### What Works:
1. **Sentence Segmentation & Vectorization**: Documents are split into semantic chunks
2. **Named Entity Recognition (NER)**: Entities are extracted from each text chunk
3. **Entity Disambiguation**: Similar entities within a document are merged
4. **Corpus Retrospective Retrieval**: Relevant text passages are retrieved for each entity
5. **Entity-Centered Sub-graph Construction**: Each entity gets its own sub-graph
6. **Intra-document Fusion**: Entity-centered sub-graphs are merged into a single KG per document

#### What's Missing:
- **Cross-Document Knowledge Graph Fusion (Section III.E)**: ❌ NOT IMPLEMENTED

## Section III.E (Knowledge Graph Fusion) - Not Implemented

The paper describes two types of fusion in Section III.E:

### 1. Entity Merging
> "Entities in the new knowledge graph may refer to the same entities in the initial knowledge graph. It is necessary to disambiguate and merge entities from the new knowledge graph with those in the initial knowledge graph."

**Status**: Not implemented for cross-document fusion

### 2. Relationship Integration
> "To obtain a more comprehensive knowledge graph, relationships in the new knowledge graph need to be integrated with those in the initial knowledge graph."

**Status**: Not implemented for cross-document fusion

### Evidence from Code

#### File: `src/kgAgent.py` (line 264)
```python
result = chain.invoke({
    "text": chunk_text,
    "target_entity": entity_dic[entity_id].get('name'),
    "related_kg": 'none'  # ← Hardcoded to 'none', no pre-existing KG loaded
})
```

The `related_kg` parameter is **hardcoded to `'none'`**, meaning:
- No previous knowledge graphs are loaded
- No cross-document entity disambiguation occurs
- No relationship integration across documents happens

### Current Behavior

When processing multiple documents (e.g., the 105 topics in MINE.json):
- Document 1 → KG1 (independent)
- Document 2 → KG2 (independent, does NOT use KG1)
- Document 3 → KG3 (independent, does NOT use KG1 or KG2)
- ...
- Document 105 → KG105 (independent, does NOT use any previous KGs)

**Result**: 105 completely independent knowledge graph files with no cross-document connections.

### What Section III.E Should Do (Per Paper Description)

Ideally, the framework should:
1. Process Document 1 → Build KG1
2. Process Document 2 with KG1 as "initial KG" → Merge into KG2
3. Process Document 3 with KG2 as "initial KG" → Merge into KG3
4. Continue accumulating knowledge across all documents

This would enable:
- Entity disambiguation across documents (e.g., recognizing "Einstein" in Doc 1 and "Albert Einstein" in Doc 2 as the same entity)
- Relationship consolidation (combining information about the same entity from multiple sources)
- A unified knowledge graph spanning all documents

### README Discrepancy

The README.md states:
> "Finally, the newly built knowledge graph is combined with the original one."

This claim is **not reflected in the current implementation**. Each document produces a standalone KG file.

## Conclusion

The current implementation successfully performs **document-level** knowledge graph construction as described in Sections III.A through III.D of the paper. However, the **cross-document knowledge graph fusion** functionality described in Section III.E appears to be:
- Described in the paper and README
- Not implemented in the released code
- Likely a planned feature or removed for simplification

### Fair Assessment?
**Yes**, it is fair to say that Section III.E (Knowledge Graph Fusion for cross-document integration) is not implemented in this repository. The code successfully constructs individual document-level KGs but does not merge them into a unified knowledge base.

---

**Date**: 2025-01-20
**Verified By**: Repository code inspection and execution testing
