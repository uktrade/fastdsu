# Security Checklist

Work through this checklist from top to bottom, ticking each item once you have confirmed it is true. Each item links to detailed guidance in the [DBT GitHub Security Policy](https://github.com/uktrade/.github/blob/main/SECURITY.md).

Last checked against the policy: _add date_

## 1. Contributor controls

Actions each contributor takes for themselves, so everyone knows what the controls are and why they exist.

- [x] [All internal contributors have read the DBT GitHub Security Policy](https://github.com/uktrade/.github/blob/main/SECURITY.md)
- [ ] [All internal contributors have completed code security training in the last year](https://github.com/uktrade/.github/blob/main/SECURITY.md#security-training)
- [x] [All internal contributors have reviewed the GitHub Safety Tips on coding in the open](https://github.com/uktrade/.github/blob/main/SECURITY.md#github-safety-tips)

## 2. Repository-level controls

Defences set up within the repository itself.

- [x] [A `.pre-commit-config.yaml` file exists so the organisation-approved hooks run before commits](https://github.com/uktrade/.github/blob/main/SECURITY.md#pre-commit-hooks)
- [x] [Repository access has been reviewed](https://github.com/uktrade/.github/blob/main/SECURITY.md#repository-access)
- [x] [A `CODEOWNERS` file exists so the right people review changes](https://github.com/uktrade/.github/blob/main/SECURITY.md#codeowners)
- [x] [The pull request template reminds reviewers to check for secrets](https://github.com/uktrade/.github/blob/main/SECURITY.md#pull-request-template)
- [x] [The mandatory custom GitHub properties are set](https://github.com/uktrade/.github/blob/main/SECURITY.md#custom-github-properties)
- [ ] [Advanced CodeQL is set up if the repository accepts PRs from forks (optional)](https://github.com/uktrade/.github/blob/main/SECURITY.md#codeql-for-fork-based-prs-optional)

## 3. Organisation-applied controls

Controls applied by an organisation administrator and verified by a repository administrator.

- [x] [The DBT GitHub security configuration is applied to the repository](https://github.com/uktrade/.github/blob/main/SECURITY.md#github-security-configuration)
- [x] [The default branch protection ruleset is applied to the default branch](https://github.com/uktrade/.github/blob/main/SECURITY.md#branch-protection-rules)
- [x] [GitHub Secret Protection is enabled and blocking secrets](https://github.com/uktrade/.github/blob/main/SECURITY.md#github-secret-protection)
- [x] [The relevant vulnerability scans are active](https://github.com/uktrade/.github/blob/main/SECURITY.md#vulnerability-scanning)