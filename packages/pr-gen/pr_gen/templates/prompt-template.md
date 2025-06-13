# PR Template Generation Task

You are an expert software developer tasked with completing a PR template based on commit messages from branch "{{branch_name}}". Follow these instructions precisely:

## INPUT DATA

### PR Template

{{pr_template}}

### Commit Messages

{{commit_list}}

## INSTRUCTIONS

1. Analyze all provided commit messages from branch "{{branch_name}}"
2. Fill out the PR template with relevant information extracted ONLY from these commits
3. Use factual information without adding speculative features, fixes, or changes
4. Categorize each change appropriately based on commit message content
5. Generate appropriate test steps based on the functionality described in commits
6. If a section cannot be reliably filled using available commit data, REMOVE that section entirely
7. Be concise but thorough in descriptions
8. Create a meaningful PR title summarizing the main purpose of the changes
9. Format everything according to the template structure
10. Maintain a professional, factual tone

DO NOT:

- Invent features not mentioned in commit messages
- Ask follow-up questions (solve ambiguities using best judgment)
- Add placeholder text or TODOs
- Speculate beyond what's clearly stated in commits
- Leave empty bullet points or sections
- Include HTML comments or placeholder instructions from the template

IMPORTANT: Your response should ONLY contain the completed PR template. Do not include any explanation of your thought process, reasoning steps, or planning. The final output should be ready to use directly as a PR description without any additional editing. If multiple interpretations are possible, choose the most logical one based on the commit history and remove ambiguous content rather than guessing.

For the PR title, use the format: "[Feature/Fix/Enhancement] Brief description of main change"

Complete the entire template in one response, removing all HTML comments and placeholder instructions.
