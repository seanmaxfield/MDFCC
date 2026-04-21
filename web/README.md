# MDFCC Web

Browser-hosted version of the MDFCC map GUI.

## Run local dev

```bash
cd web
npm install
npm run dev
```

## Build

```bash
cd web
npm run build
```

## GitHub Pages deploy

Deployment is automated via `.github/workflows/web-pages.yml`.

1. Open repository Settings -> Pages
2. Set Source to **GitHub Actions**
3. Push changes under `web/**` to `main`
