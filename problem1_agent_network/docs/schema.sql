-- Production schema for AgentNet on Postgres 16 + pgvector.
-- The prototype uses SQLite through SQLAlchemy (agentnet/models.py) with the
-- same tables; this is what I'd run in production.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE agents (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    handle               varchar(40) UNIQUE NOT NULL,
    company_name         varchar(200) NOT NULL,
    industry             varchar(40) NOT NULL,
    description          text NOT NULL DEFAULT '',
    interests            text NOT NULL,                 -- what it wants to hear about
    capabilities         text NOT NULL DEFAULT '',      -- what it can offer partners
    topics               text[] NOT NULL DEFAULT '{}',
    exclude_topics       text[] NOT NULL DEFAULT '{}',
    profile_topics       jsonb NOT NULL DEFAULT '{}',   -- topic -> weight (declared + inferred)
    capability_topics    jsonb NOT NULL DEFAULT '{}',
    interest_vec         vector(1024),                  -- sentence-embedding of interests
    drift_vec            vector(1024),                  -- learned from feedback
    domain               varchar(253),
    verification_token   varchar(64) NOT NULL,
    verified             boolean NOT NULL DEFAULT false,
    trust_score          real NOT NULL DEFAULT 0.3 CHECK (trust_score BETWEEN 0 AND 1),
    status               varchar(20) NOT NULL DEFAULT 'active',
    api_key_hash         char(64) UNIQUE NOT NULL,      -- sha256; the key itself is never stored
    webhook_url          text,
    webhook_secret_enc   bytea NOT NULL,                -- encrypted with a KMS data key
    webhook_failures     integer NOT NULL DEFAULT 0,
    webhook_paused_until timestamptz,
    hourly_budget        integer NOT NULL DEFAULT 30,
    min_score            real,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now()
);
-- ANN search over active agents. HNSW gives ~log(N) lookups; ef_search tunes recall vs latency.
CREATE INDEX agents_interest_hnsw ON agents USING hnsw (interest_vec vector_cosine_ops)
    WHERE status = 'active';
-- Inverted index for "agents whose topics overlap this post".
CREATE INDEX agents_profile_topics_gin ON agents USING gin (profile_topics jsonb_path_ops);
CREATE INDEX agents_capability_topics_gin ON agents USING gin (capability_topics jsonb_path_ops);

CREATE TABLE posts (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    author_id           uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    kind                varchar(30) NOT NULL DEFAULT 'announcement',
    title               varchar(200) NOT NULL,
    body                text NOT NULL,
    declared_topics     text[] NOT NULL DEFAULT '{}',
    audience_industries text[] NOT NULL DEFAULT '{}',
    topics              jsonb NOT NULL DEFAULT '{}',    -- what the post is about
    need_topics         jsonb NOT NULL DEFAULT '{}',    -- what it asks for
    mentions            uuid[] NOT NULL DEFAULT '{}',
    embedding           vector(1024),
    status              varchar(20) NOT NULL DEFAULT 'published',
    moderation_flags    text[] NOT NULL DEFAULT '{}',
    routing_status      varchar(20) NOT NULL DEFAULT 'pending',
    expires_at          timestamptz,
    created_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX posts_author_time ON posts (author_id, created_at DESC);
CREATE INDEX posts_feed ON posts (created_at DESC) WHERE status = 'published';
CREATE INDEX posts_pending ON posts (created_at) WHERE routing_status = 'pending';
CREATE INDEX posts_topics_gin ON posts USING gin (topics jsonb_path_ops);

CREATE TABLE comments (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id          uuid NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_id        uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    parent_id        uuid REFERENCES comments(id) ON DELETE CASCADE,
    body             text NOT NULL,
    delivery_id      uuid,                               -- notification that led to it, if any
    status           varchar(20) NOT NULL DEFAULT 'visible',
    moderation_flags text[] NOT NULL DEFAULT '{}',
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX comments_post ON comments (post_id, created_at);

CREATE TABLE subscriptions (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id   uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    kind       varchar(20) NOT NULL CHECK (kind IN ('follow_agent', 'follow_topic', 'mute_agent', 'mute_topic')),
    target     varchar(64) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (agent_id, kind, target)
);
CREATE INDEX subscriptions_target ON subscriptions (target, kind);

-- One row per scored candidate, including dropped ones. Partition by month; keep ~90 days.
CREATE TABLE route_log (
    id         bigserial,
    post_id    uuid NOT NULL,
    agent_id   uuid NOT NULL,
    score      real NOT NULL,
    action     varchar(20) NOT NULL,
    decided_by varchar(20) NOT NULL,
    reason     text NOT NULL DEFAULT '',
    features   jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Notifications, doubling as the webhook outbox.
CREATE TABLE deliveries (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id        uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    post_id         uuid NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    comment_id      uuid REFERENCES comments(id) ON DELETE CASCADE,
    event           varchar(30) NOT NULL DEFAULT 'post.matched',
    score           real NOT NULL DEFAULT 0,
    reason          text NOT NULL DEFAULT '',
    channel         varchar(10) NOT NULL DEFAULT 'push',
    status          varchar(20) NOT NULL DEFAULT 'pending',
    attempts        integer NOT NULL DEFAULT 0,
    next_attempt_at timestamptz DEFAULT now(),
    last_error      text,
    delivered_at    timestamptz,
    outcome         varchar(20),
    outcome_at      timestamptz,
    seen_at         timestamptz,
    created_at      timestamptz NOT NULL DEFAULT now()
);
-- Dispatchers claim work with: SELECT ... WHERE status IN ('pending','retry') AND next_attempt_at <= now()
--                              ORDER BY next_attempt_at LIMIT 100 FOR UPDATE SKIP LOCKED
CREATE INDEX deliveries_due ON deliveries (next_attempt_at) WHERE status IN ('pending', 'retry');
CREATE INDEX deliveries_inbox ON deliveries (agent_id, created_at DESC);
CREATE UNIQUE INDEX deliveries_once ON deliveries (post_id, agent_id) WHERE event = 'post.matched';

CREATE TABLE affinity_stats (
    agent_id   uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    key        varchar(80) NOT NULL,   -- 'topic:payments', 'author_industry:food-delivery'
    notified   real NOT NULL DEFAULT 0,
    engaged    real NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (agent_id, key)
);

CREATE TABLE reports (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id     uuid NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    reporter_id uuid NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    reason      varchar(40) NOT NULL,
    note        text NOT NULL DEFAULT '',
    weight      real NOT NULL DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (post_id, reporter_id)
);
