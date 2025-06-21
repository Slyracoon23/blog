# Claude PR Blog System Instructions

## Overview
This system helps write detailed technical blogs about open source Pull Requests (PRs), specifically optimized for vLLM but adaptable to other repositories. It uses a main agent + sub-agent architecture where:

- **Main Agent**: Coordinates PR review commands and updates the main blog index
- **Sub-Agents**: Handle individual PR analysis, data fetching, and blog writing

## 🚨 CRITICAL REQUIREMENTS

### PR File Location & Linking
- **ALL PR files MUST be in `opensource/pr/{REPO_NAME}/` directory**
- **NEVER create PR files in root `opensource/` directory**  
- **Files MUST be includable via `{{< include pr/{repo_name}/filename.qmd >}}` macro**
- **Filename format: `pr_{NUMBER}_{sanitized_title}.qmd`**
- **Full path format: `opensource/pr/{repo_name}/pr_{NUMBER}_{sanitized_title}.qmd`**

### Include Macro Pattern
```markdown
::: {.callout-warning collapse="true" title="🐛 Bug Fix - PR #19561: Don't attempt to use triton if no driver is active"}
{{< include pr/vllm/pr_19561_triton_driver_check.qmd >}}
:::
```

## System Architecture

### Main Agent Responsibilities
1. **Command Processing**: Parse user commands about which PRs to analyze
2. **Sub-Agent Coordination**: Create and manage sub-agents for each PR
3. **Index Management**: Update main blog files with new PR entries
4. **Summary Reporting**: Aggregate results from all sub-agents

### Sub-Agent Responsibilities  
1. **Data Collection**: Use MCP GitHub tools to fetch PR details, diffs, and comments
2. **Content Synthesis**: Create technical blog following vLLM learning guide specs
3. **File Generation**: Write individual `.qmd` files in `opensource/pr/{repo_name}/` directory
4. **Status Reporting**: Return summary to main agent

## Command Format
```
ANALYZE_PRS: pr_number1, pr_number2, pr_number3
REPO: owner/repo_name (optional, defaults to vllm-project/vllm)
```

Example:
```
ANALYZE_PRS: 19561, 19642, 19588
REPO: vllm-project/vllm
```

## Implementation Flow

### When User Issues Command
```
User: "ANALYZE_PRS: 19642, 19588"
```

**Main Agent Actions:**
1. Parse command → Extract PR numbers [19642, 19588]
2. Set default repo → vllm-project/vllm
3. Create sub-agent for each PR
4. Monitor progress and collect results
5. Update blog index with new entries

**Sub-Agent Actions (in parallel):**
Each sub-agent follows the 4-phase workflow below.

## Sub-Agent Workflow

### 1. Data Collection Phase
For each PR, the sub-agent must:

```python
# Required MCP Tool Calls (in parallel for efficiency)
1. mcp_github_get_pull_request(owner, repo, pullNumber)
2. mcp_github_get_pull_request_diff(owner, repo, pullNumber) 
3. mcp_github_get_pull_request_comments(owner, repo, pullNumber)
4. mcp_github_get_pull_request_reviews(owner, repo, pullNumber)
5. mcp_github_get_pull_request_files(owner, repo, pullNumber)
```

### 2. Content Synthesis Phase
Create blog following this **exact structure** (based on vLLM cursor rules):

```markdown
## [#{PR_NUMBER}]({PR_URL}) - {PR_TITLE}

#### Overview
- Concise problem/feature summary
- Motivation & architectural context
- Key technical background

#### Code Changes (Verified)
For each changed file:
**File**: `path/to/file.py`

```diff
@@ -old +new @@
- old line
+ new line
```

**Explanation**: 
- Line-by-line breakdown of WHY changes were made
- Performance implications and trade-offs
- Links to related modules

#### PR Discussion & Comments
**@reviewer → @author** — Summary of discussion point
"Exact quote (≤30 words) from key comment"

#### Key Takeaways
- Why the change matters
- What learners should remember
- Benchmarks/empirical results

#### Further Reading
- Official docs links
- Papers/blog posts
- Direct PR link
```

### 3. File Generation Phase
- **CRITICAL**: Generate filename: `pr_{PR_NUMBER}_{sanitized_title}.qmd`
- **CRITICAL**: Write file to `opensource/pr/{repo_name}/` directory (MUST be in `/pr/{repo_name}` subfolder)
- **CRITICAL**: Ensure proper Quarto metadata headers
- **CRITICAL**: File must be includable via `{{< include pr/{repo_name}/filename.qmd >}}` macro

### 4. Reporting Phase
Return to main agent:
```json
{
  "pr_number": "19561",
  "filename": "pr_19561_triton_driver_check.qmd", 
  "title": "Don't attempt to use triton if no driver is active",
  "summary": "Fixed Triton driver validation preventing crashes on non-GPU platforms",
  "categories": ["bug-fix", "triton", "gpu-drivers"],
  "key_highlights": [
    "Cross-platform compatibility fix",
    "Graceful fallback implementation", 
    "Robust error handling"
  ],
  "file_path": "opensource/pr/vllm/pr_19561_triton_driver_check.qmd"
}
```

## Main Agent Integration Tasks

### Update Main Blog Index
After receiving sub-agent reports, update `opensource.qmd`:

1. **Add to listing metadata** (if new categories/tags)
2. **Update repository descriptions** (if new repos analyzed)

```markdown
### Recent PR Analyses
- [#{PR_NUM} - {TITLE}](opensource/pr/{repo_name}/{FILENAME}) - {BRIEF_SUMMARY}
```

### Include Macro Usage Pattern
For main blog posts that reference multiple PRs, use the include macro:

```markdown
::: {.callout-warning collapse="true" title="🐛 Bug Fix - PR #{PR_NUMBER}: {PR_TITLE}"}
{{< include pr/{repo_name}/pr_{PR_NUMBER}_{sanitized_title}.qmd >}}
:::
```

**Example from weekly summary blog:**
```markdown
::: {.callout-warning collapse="true" title="🐛 Bug Fix - PR #19561: Don't attempt to use triton if no driver is active"}
{{< include pr/vllm/pr_19561_triton_driver_check.qmd >}}
:::
```

### Update Blog Metadata
Create or update `opensource/_metadata.yml` with new entries:

```yaml
- path: pr/vllm/pr_19561_triton_driver_check.qmd
  title: "#19561 - Don't attempt to use triton if no driver is active"
  description: "Fixed Triton driver validation preventing crashes on non-GPU platforms"
  categories: [bug-fix, triton, gpu-drivers]
  date: "2025-01-15"
  image: images/thumbnail_template.jpg
```

## Error Handling

### Sub-Agent Failures
- **PR Not Found**: Log error, continue with other PRs
- **API Rate Limits**: Implement exponential backoff
- **Malformed Data**: Validate all fetched data before processing

### Main Agent Failures  
- **File Write Errors**: Ensure directory permissions
- **Blog Update Conflicts**: Check for concurrent edits
- **Invalid Commands**: Provide clear error messages and examples

## Quality Assurance

### Content Standards
- **Technical Accuracy**: Verify all code diffs and explanations
- **Clarity**: Write for developers learning vLLM internals
- **Completeness**: Include all required sections per cursor rules
- **Consistency**: Follow established naming and structure patterns

### File Standards
- **Naming Convention**: `pr_{NUMBER}_{sanitized_title}.qmd`
- **Directory Structure**: All PRs MUST be in `opensource/pr/{repo_name}/` (never in root opensource/)
- **Include Macro Compatibility**: Files must work with `{{< include pr/{repo_name}/filename.qmd >}}`
- **Metadata Compliance**: Proper Quarto front matter
- **Link Validation**: Ensure all external links work

## Main Blog Structure Pattern

Based on the vLLM weekly blog format, the main blog should follow this structure:

```markdown
# {REPO} Merged PRs - Week ({DATE_RANGE})

**Total PRs: {COUNT}**

## Summary by Category
- **Bug Fixes**: X PRs (Y%)
- **Performance & Optimization**: X PRs (Y%)
- **Model Support**: X PRs (Y%)
- etc.

## Key Highlights
1. **Major Focus**: Description
2. **Performance**: Description
3. **Architecture**: Description

---

::: {.panel-tabset}

## 🐛 Bug Fixes ({COUNT} PRs)

::: {.callout-warning collapse="true" title="🐛 Bug Fix - PR #{NUMBER}: {TITLE}"}
{{< include pr/{repo_name}/pr_{NUMBER}_{sanitized_title}.qmd >}}
:::

{Additional PRs can be listed as simple links or included}

## ⚡ Performance & Optimization ({COUNT} PRs)

{Similar structure for each category}

:::
```

## Success Metrics
- **Completeness**: All required sections present
- **Accuracy**: Code diffs match actual PR changes  
- **Readability**: Clear explanations for target audience
- **Timeliness**: Efficient parallel processing
- **Maintainability**: Consistent structure for future updates
