---
name: research-company
description: Research a company with multi-source verification and taxonomy-based feature tracking
arguments:
  - name: company
    description: "Company name to research (e.g., 'Bloomerang', 'Stripe')"
    required: true
  - name: taxonomy
    description: "Taxonomy name to use for feature matching (e.g., 'k12-fundraising')"
    required: false
  - name: context
    description: "Optional context to help focus research (e.g., 'donor management', 'payments')"
    required: false
---

# Multi-Source Company Research: $ARGUMENTS.company

Research **$ARGUMENTS.company** using multiple source types for verified feature discovery.

## Research Workflow Overview

This workflow gathers evidence from multiple sources (help docs, marketing, reviews) to build a verified competitor profile with taxonomy-linked features.

---

## Step 1: Check Prerequisites

First, check if a taxonomy exists (if specified):

```bash
# If taxonomy argument provided, verify it exists
toolhub taxonomy list --format json
```

If taxonomy is specified but doesn't exist, ask user:
- Create the taxonomy now
- Proceed without taxonomy (features tracked as freeform)
- Cancel

---

## Step 2: Identify Sources

Search for 2-3 source types for **$ARGUMENTS.company**:

| Source Type | Search Query | Priority |
|-------------|--------------|----------|
| Help Docs | "$ARGUMENTS.company help center knowledge base support docs" | High |
| Marketing | "$ARGUMENTS.company features product $ARGUMENTS.context" | Medium |
| Reviews | "$ARGUMENTS.company G2 reviews" OR "$ARGUMENTS.company Capterra reviews" | Medium |

Use WebSearch to find URLs for each source type. Target:
- 1 help docs URL (e.g., support.company.com, help.company.com)
- 1 marketing URL (e.g., company.com/features, company.com/product)
- 1 review URL (e.g., g2.com/products/company, capterra.com/software/company)

---

## Step 3: Register Sources

For each URL found, create a source record:

```bash
# Help docs source
toolhub source add "https://support.company.com" \
  --type help_docs \
  --collection "$ARGUMENTS.company-research" \
  --format json

# Marketing source
toolhub source add "https://company.com/features" \
  --type marketing \
  --collection "$ARGUMENTS.company-research" \
  --format json

# Review source
toolhub source add "https://www.g2.com/products/company" \
  --type review \
  --collection "$ARGUMENTS.company-research" \
  --format json
```

**Capture the source IDs** from JSON output - you'll need them for evidence.

---

## Step 4: Extract Features Per Source

For each source URL, use WebFetch to read the content and extract features.

### Extraction Format

For each source, extract a list of features with supporting quotes:

```
Source: [URL] (source_id: [ID])
Type: [help_docs|marketing|review]

Features Found:
1. [feature-name]: "[exact quote supporting this feature]"
2. [feature-name]: "[exact quote supporting this feature]"
...
```

### Extraction Guidelines

- **Help Docs**: Look for article titles, feature sections, how-to guides
- **Marketing**: Look for feature lists, product descriptions, comparison tables
- **Reviews**: Look for mentioned capabilities, praised features, use cases

---

## Step 5: Taxonomy Reconciliation

For each extracted feature, match against existing taxonomy or prompt for new:

### If Taxonomy Exists

```bash
# Get current taxonomy structure
toolhub taxonomy show "$ARGUMENTS.taxonomy" --format json
```

For each feature found:
1. **Fuzzy match** against existing feature labels in the taxonomy
2. **If match found**: Use the existing feature key
3. **If no match**: Ask user which group to add it to:

```
Feature "Email Integration" not found in taxonomy.

Which group should this feature belong to?
1. donor-management (Donor Management)
2. online-giving (Online Giving)
3. communications (Communications)
4. Create new group
5. Skip this feature
```

If adding new feature:
```bash
toolhub taxonomy add-feature "$ARGUMENTS.taxonomy" "[group-key]" \
  --key "[feature-key]" \
  --label "[Feature Label]"
```

### If No Taxonomy

Track features as freeform strings. Later, run:
```bash
toolhub taxonomy create "[domain-name]" --domain "[Domain Label]"
```

---

## Step 6: Create Competitor Entity

Create or update the competitor entity with initial profile:

```bash
# Check if competitor exists
toolhub entity list --type competitor --name "$ARGUMENTS.company" --format json

# Create new competitor
toolhub entity create competitor "$ARGUMENTS.company" \
  --profile '{
    "description": "[1-2 sentence description]",
    "website": "[company website]",
    "features": {},
    "funding": {
      "total": null,
      "rounds": []
    }
  }' \
  --collection "research" \
  --format json
```

**Capture the entity ID** from output.

---

## Step 7: Record Evidence

For each feature found, add evidence linking the quote to the source:

```bash
toolhub evidence add "[entity-id]" "features.[feature-key]" \
  --source "[source-id]" \
  --quote "[exact quote from source]" \
  --confidence 0.9
```

Repeat for each (feature, source, quote) tuple found in Step 4.

---

## Step 8: Calculate Source Counts

After all evidence is recorded, count distinct sources per feature:

```bash
# List evidence for this competitor
toolhub evidence list --entity "[entity-id]" --format json
```

Build feature map with source counts:
```json
{
  "feature-key-1": { "sources": 2 },
  "feature-key-2": { "sources": 3 },
  "feature-key-3": { "sources": 1 }
}
```

Features with 2+ sources are **verified** (✓✓).

---

## Step 9: Update Competitor Profile

Update the competitor with derived source counts:

```bash
toolhub entity update "[entity-id]" \
  --profile '{
    "description": "[description]",
    "website": "[website]",
    "features": {
      "feature-key-1": { "sources": 2 },
      "feature-key-2": { "sources": 3 },
      "feature-key-3": { "sources": 1 }
    },
    "funding": { "total": null, "rounds": [] }
  }'
```

---

## Step 10: Present Results

Show the final research summary:

```markdown
## $ARGUMENTS.company Research Complete

**Description:** [description]
**Website:** [website]

### Verified Features (2+ sources)
| Feature | Sources | Evidence |
|---------|---------|----------|
| [label] | ✓✓ (3) | help_docs, marketing, review |
| [label] | ✓✓ (2) | help_docs, review |

### Unverified Features (1 source)
| Feature | Sources | Evidence |
|---------|---------|----------|
| [label] | ✓ (1) | marketing |

### Sources Analyzed
1. [help_docs] https://support.company.com
2. [marketing] https://company.com/features
3. [review] https://g2.com/products/company

### Evidence Recorded
- [X] evidence records linking features to sources

To generate a feature matrix report:
toolhub report generate competitor-feature-matrix --collection research
```

---

## Error Handling

- **No help docs found**: Proceed with marketing and reviews only
- **No taxonomy specified**: Track features as freeform, suggest creating taxonomy later
- **Entity already exists**: Ask user to update or skip
- **WebFetch fails**: Note the source as unavailable, continue with others
- **Feature ambiguous**: Ask user to clarify or skip

---

## Example Session

**Input:** `/toolhub:research-company Bloomerang --taxonomy k12-fundraising --context "donor management"`

**Output flow:**
1. Found sources: support.bloomerang.com, bloomerang.com/features, g2.com/products/bloomerang
2. Registered 3 sources in toolhub
3. Extracted features: donor-profiles, giving-history, recurring-gifts, email-integration
4. Matched 3 features to taxonomy, added 1 new (email-integration → communications)
5. Created competitor entity, recorded 8 evidence records
6. Final profile: 2 verified features, 2 unverified

```
## Bloomerang Research Complete

**Description:** Donor management platform for small nonprofits
**Website:** https://bloomerang.com

### Verified Features (2+ sources)
| Feature | Sources | Evidence |
|---------|---------|----------|
| Donor Profiles | ✓✓ (3) | help_docs, marketing, review |
| Giving History | ✓✓ (2) | help_docs, marketing |

### Unverified Features (1 source)
| Feature | Sources | Evidence |
|---------|---------|----------|
| Recurring Gifts | ✓ (1) | marketing |
| Email Integration | ✓ (1) | review |

### Sources Analyzed
1. [help_docs] https://support.bloomerang.com
2. [marketing] https://bloomerang.com/features
3. [review] https://g2.com/products/bloomerang

### Evidence Recorded
- 8 evidence records linking features to sources
```
