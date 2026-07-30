# Deploying Beeline to AWS

## Live deployment

    App      https://beeline-633626894967-us-west-2.s3.us-west-2.amazonaws.com/app/index.html
    API      https://siy3v64bjw.us-west-2.awsapprunner.com
    Region   us-west-2 (Oregon)

us-west-2 rather than us-west-1 despite being closer to SF: App Runner has no
endpoint in N. California at all. Latency difference is a few milliseconds.

The shape, and why:

| Piece | Where | Why |
|---|---|---|
| 96 cut clips (~512MB) | S3 + CloudFront | Static bytes with range requests. A container is the wrong place to serve video from |
| Frontend (~380KB) | S3 + CloudFront | Static build |
| FastAPI | App Runner (ECR image) | HTTPS, autoscaling and deploys without a cluster to manage |
| Concept graph | Neo4j Aura | Already hosted; nothing moves |
| Credentials | Secrets Manager | `.env` is gitignored and must never enter the image |

The deployed API runs with `BEELINE_CANNED=1`. Gap filling shells out to yt-dlp
and ffmpeg, which takes 30–90 seconds per fill and gets throttled from
datacentre IPs — fine on a laptop, not something to do live on stage. Fill the
gaps locally, ship the clips, and serve the cached responses.

## Before you start

```bash
aws configure                 # or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
export AWS_REGION=us-east-1
```

Your `.env` must be populated, the clips cut (`beeline/ingestion/cut.py`), and
the demo queries warmed against the API so `data/cache/api/` is not empty —
in canned mode, an uncached query returns 503 rather than a wrong answer.

## Deploy

If `docker build` fails with `failed to set up container networking`, the daemon
cannot create veth pairs in your environment; build with host networking:

```bash
docker build --network=host -t beeline:latest .
```

```bash
./deploy/deploy.sh media       # clips  -> S3
./deploy/deploy.sh secrets     # .env   -> Secrets Manager (values never printed)
./deploy/deploy.sh api         # image  -> ECR
```

Then create the App Runner service once, in the console — it needs an access
role and the secret ARN, which are easier to review there than to conjure from a
script:

- **Source**: the ECR image `beeline:latest`, deploy on push
- **Port**: 8000
- **Health check**: HTTP `/api/health`
- **Environment**:
  - `BEELINE_STORE=neo4j`
  - `BEELINE_CANNED=1`
  - `MEDIA_BASE_URL=https://<your-cloudfront-domain>`
- **Secrets** (from `beeline/env`): `OPENAI_API_KEY`, `NEO4J_URI`,
  `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `TWELVELABS_API_KEY`, `TWELVELABS_INDEX_ID`

Then a CloudFront distribution over the bucket, with two behaviours: `/media/*`
and `/app/*`, origin access control on, default root object `app/index.html`.

Finally, rebuild the frontend so it points at the live API:

```bash
BEELINE_API_URL=https://<app-runner-url> ./deploy/deploy.sh frontend
```

## Verified locally

The image was built and run against the live Aura instance before shipping:

```
health          canned_only: true, store: neo4j
cached query    13 clips, 26.9m, external fills intact
media_url       https://cdn.example.test/media/v6_c8.mp4   (CDN rewrite)
uncached query  HTTP 503                                   (refuses, does not guess)
graph           97 concepts, 181 edges
```

`clips_cut: 0` inside the container is expected -- the clips are deliberately
not in the image.

## Costs

App Runner is roughly $25/month at idle — it does not scale to zero. S3 and
CloudFront for 512MB and demo traffic are cents. **Delete the App Runner service
when the demo is over**, or it bills indefinitely.

## Gotchas

- `MEDIA_BASE_URL` is applied to responses on the way out, and cached responses
  always store the relative path. So you can point at a different CDN without
  invalidating the cache.
- Re-pruning the payload means reloading Aura and clearing `data/cache/api/`, or
  you are serving history. This bit us locally more than once.
- CloudFront caches `index.html` unless told not to; the deploy script sets
  `no-cache` on it and immutable on everything else.
