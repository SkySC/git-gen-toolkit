# Conventional Commit Format

[type] subject

body

footer

## Types

- [feat] - A new feature
- [fix] - A bug fix
- [docs] - Documentation changes
- [style] - Changes that don't affect code behavior (formatting, etc.)
- [refactor] - Code changes that neither fix a bug nor add a feature
- [perf] - Performance improvements
- [test] - Adding or fixing tests
- [build] - Changes to build system or dependencies
- [ci] - Changes to CI configuration
- [chore] - Other changes that don't modify src or test files

## Examples

[feat] add user authentication system

Implement OAuth2 authentication for users to securely log in
with their existing Google and GitHub accounts.

Refs: #123

---

[fix] correct calculation in tax module

The tax rate was being applied incorrectly when special
conditions were met.

Fixes: #456
