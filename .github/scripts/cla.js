// CLA check for pull requests, run by .github/workflows/cla.yml through actions/github-script.
//
// A pull request passes when every author of its commits, and its opener, has accepted CLA.md or
// has write access to the repository. Contributors accept by commenting SIGN_PHRASE on a pull
// request; the acceptance is recorded (GitHub user id, login, time, link to the comment) in
// SIGNATURES_PATH on the SIGNATURES_BRANCH branch, which is created on first use. The result is a
// commit status named STATUS_CONTEXT on the pull request head: make it a required check.
//
// This script runs in the base repository's context and never checks out or executes pull
// request code.

const CLA_VERSION = 'v1.0';
const SIGN_PHRASE = 'I have read the Growgraph CLA and I accept it.';
const RECHECK_PHRASE = 'recheck cla';
const SIGNATURES_BRANCH = 'cla-signatures';
const SIGNATURES_PATH = `signatures/cla-${CLA_VERSION}.json`;
const STATUS_CONTEXT = 'cla';
const MARKER = '<!-- growgraph-cla-check -->';
const EXEMPT_ROLES = new Set(['admin', 'maintain', 'write']);

function normalize(text) {
  return text.trim().replace(/\s+/g, ' ').replace(/\.$/, '').toLowerCase();
}

function isBot(user) {
  return user.type === 'Bot' || user.login.endsWith('[bot]');
}

async function loadSignatures(github, owner, repo) {
  try {
    const { data } = await github.rest.repos.getContent({
      owner, repo, path: SIGNATURES_PATH, ref: SIGNATURES_BRANCH,
    });
    const list = JSON.parse(Buffer.from(data.content, 'base64').toString('utf8'));
    return { sha: data.sha, list };
  } catch (error) {
    if (error.status === 404) return { sha: undefined, list: [] };
    throw error;
  }
}

async function ensureSignaturesBranch(github, owner, repo) {
  try {
    await github.rest.repos.getBranch({ owner, repo, branch: SIGNATURES_BRANCH });
  } catch (error) {
    if (error.status !== 404) throw error;
    const { data: repository } = await github.rest.repos.get({ owner, repo });
    const { data: base } = await github.rest.git.getRef({
      owner, repo, ref: `heads/${repository.default_branch}`,
    });
    await github.rest.git.createRef({
      owner, repo, ref: `refs/heads/${SIGNATURES_BRANCH}`, sha: base.object.sha,
    });
  }
}

async function recordSignature(github, owner, repo, comment, pullRequest) {
  const user = comment.user;
  await ensureSignaturesBranch(github, owner, repo);
  // Two acceptances at once race on the file's sha; the loser re-reads and retries.
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const { sha, list } = await loadSignatures(github, owner, repo);
    if (list.some((entry) => entry.id === user.id)) return;
    list.push({
      login: user.login,
      id: user.id,
      cla: CLA_VERSION,
      signed_at: comment.created_at,
      pull_request: pullRequest.html_url,
      comment: comment.html_url,
    });
    try {
      await github.rest.repos.createOrUpdateFileContents({
        owner,
        repo,
        branch: SIGNATURES_BRANCH,
        path: SIGNATURES_PATH,
        message: `CLA ${CLA_VERSION}: accepted by @${user.login}`,
        content: Buffer.from(`${JSON.stringify(list, null, 2)}\n`).toString('base64'),
        sha,
      });
      return;
    } catch (error) {
      if (error.status !== 409 && error.status !== 422) throw error;
    }
  }
  throw new Error(`Could not record the CLA acceptance of @${user.login}`);
}

async function hasWriteAccess(github, owner, repo, login) {
  try {
    const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
      owner, repo, username: login,
    });
    return EXEMPT_ROLES.has(data.role_name) || EXEMPT_ROLES.has(data.permission);
  } catch (error) {
    if (error.status === 404) return false;
    throw error;
  }
}

// Who must accept: commit authors with a GitHub account, plus the opener. Commits whose author
// e-mail is not linked to any GitHub account cannot be matched, so they block the check.
async function requiredSigners(github, owner, repo, pullRequest) {
  const commits = await github.paginate(github.rest.pulls.listCommits, {
    owner, repo, pull_number: pullRequest.number, per_page: 100,
  });
  const users = new Map();
  const unlinked = new Set();
  for (const commit of commits) {
    if (commit.author) {
      if (!isBot(commit.author)) users.set(commit.author.id, commit.author.login);
    } else {
      const { name, email } = commit.commit.author;
      unlinked.add(`${name} <${email}>`);
    }
  }
  if (!isBot(pullRequest.user)) users.set(pullRequest.user.id, pullRequest.user.login);
  return { users, unlinked };
}

function commentBody(claUrl, missing, unlinked) {
  if (missing.length === 0 && unlinked.length === 0) {
    return `${MARKER}\nAll contributors to this pull request have accepted the [Growgraph CLA](${claUrl}). Thank you!`;
  }
  const lines = [
    MARKER,
    '### Contributor License Agreement',
    '',
    `Thank you for your contribution! Before we can merge it, each contributor needs to accept the [Growgraph Contributor License Agreement](${claUrl}) once. You keep the copyright in your work, and it stays available under an open source license.`,
  ];
  if (missing.length > 0) {
    lines.push(
      '',
      `**Still to accept:** ${missing.map((login) => `@${login}`).join(', ')}`,
      '',
      'To accept, read the agreement and post this comment on this pull request:',
      '',
      '```',
      SIGN_PHRASE,
      '```',
      '',
      'If you write this as part of your job or for a client, make sure they allow it; see section 4(c) and section 11 of the agreement.',
    );
  }
  if (unlinked.length > 0) {
    lines.push(
      '',
      `**Commits we cannot match to a GitHub account:** ${unlinked.map((author) => `\`${author}\``).join(', ')}.`,
      'Add that e-mail address to your GitHub account (or amend the commits to use one that is), then push again.',
    );
  }
  lines.push('', `<sub>Comment \`${RECHECK_PHRASE}\` to run this check again.</sub>`);
  return lines.join('\n');
}

async function upsertComment(github, owner, repo, number, body, passed) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner, repo, issue_number: number, per_page: 100,
  });
  const existing = comments.find((comment) => comment.body && comment.body.includes(MARKER));
  if (existing) {
    if (existing.body !== body) {
      await github.rest.issues.updateComment({ owner, repo, comment_id: existing.id, body });
    }
  } else if (!passed) {
    // Nobody needs telling when everyone was already covered.
    await github.rest.issues.createComment({ owner, repo, issue_number: number, body });
  }
}

async function evaluate(github, owner, repo, pullRequest) {
  const { data: repository } = await github.rest.repos.get({ owner, repo });
  const claUrl = `https://github.com/${owner}/${repo}/blob/${repository.default_branch}/CLA.md`;
  const { users, unlinked } = await requiredSigners(github, owner, repo, pullRequest);
  const { list } = await loadSignatures(github, owner, repo);
  const signed = new Set(list.map((entry) => entry.id));

  const missing = [];
  for (const [id, login] of users) {
    if (signed.has(id)) continue;
    if (await hasWriteAccess(github, owner, repo, login)) continue;
    missing.push(login);
  }
  const passed = missing.length === 0 && unlinked.size === 0;
  const pending = missing.length + unlinked.size;

  await github.rest.repos.createCommitStatus({
    owner,
    repo,
    sha: pullRequest.head.sha,
    context: STATUS_CONTEXT,
    state: passed ? 'success' : 'failure',
    description: passed
      ? 'All contributors have accepted the CLA'
      : `${pending} contributor${pending === 1 ? '' : 's'} still to accept the CLA`,
    target_url: claUrl,
  });
  await upsertComment(
    github, owner, repo, pullRequest.number,
    commentBody(claUrl, missing, [...unlinked]), passed,
  );
  return passed;
}

module.exports = async ({ github, context, core }) => {
  const { owner, repo } = context.repo;
  let pullRequest;

  if (context.eventName === 'pull_request_target') {
    pullRequest = context.payload.pull_request;
  } else if (context.eventName === 'issue_comment') {
    const { issue, comment } = context.payload;
    if (!issue.pull_request || isBot(comment.user)) return;
    const text = normalize(comment.body);
    const signing = text === normalize(SIGN_PHRASE);
    if (!signing && text !== normalize(RECHECK_PHRASE)) return;
    ({ data: pullRequest } = await github.rest.pulls.get({
      owner, repo, pull_number: issue.number,
    }));
    if (signing) {
      await recordSignature(github, owner, repo, comment, pullRequest);
      core.info(`Recorded CLA ${CLA_VERSION} acceptance by @${comment.user.login}`);
    }
  } else {
    return;
  }

  const passed = await evaluate(github, owner, repo, pullRequest);
  core.info(passed ? 'CLA check passed' : 'CLA check pending signatures');
};
