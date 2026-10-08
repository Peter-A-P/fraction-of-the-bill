# Putting the website on smallmodel.peterparker.ca

The website is `site/`: one static page, its stylesheet, two scripts, two fonts and
`results.json`. There is no backend, so hosting it is a file upload and one DNS record. It
follows the same path as project 12's `finishline.peterparker.ca` and project 08's
`capacity.peterparker.ca`, in the style of the other project pages on peterparker.ca.

**Every number on the page comes from `site/results.json`**, and nothing on the page is
typed by hand. `smallprint site` writes the file from the same runs, load tests and
break-even arithmetic as the README's tables (`smallprint/site.py`), and the gate decisions
from `gate-runs/`. The runs are not in the repository, so the file is written on the machine
that has them and committed; the deploy uploads `site/` as it is and builds nothing. In CI,
`tests/test_site.py` checks every curve in the committed file against the Python break-even
and runs the calculator's `site/breakeven.js` under Node against it over a grid of inputs, so
neither the file nor the script can drift from the code the tables come from.

When the results change: `smallprint report ...` for the tables as before, then
`smallprint site --build-dir data/build/full`, and commit `site/results.json` with them.

Steps 1 to 3 are done once, by Peter, in the Azure portal, the Cloudflare dashboard and
GitHub. Step 4 is automatic after that.

## 0. Check the subscription before creating anything

Make sure the Azure CLI is signed in to the subscription the site belongs in; a CLI can
hold more than one login. `<subscription>` and `<resource-group>` below are yours to fill
in.

```powershell
az login
az account set --subscription <subscription>
az staticwebapp list -o table              # the subscription's existing apps
```

## 1. Create the Static Web App

Portal: [Azure Portal](https://portal.azure.com), Create a resource, **Static Web App**
([Microsoft's walkthrough](https://learn.microsoft.com/en-us/azure/static-web-apps/get-started-portal)).

| Setting | Value |
|---|---|
| Subscription | `<subscription>` |
| Resource group | `<resource-group>` |
| Name | `smallmodel-peterparker-ca` |
| Plan type | **Free** |
| Region | any near you; content is served from a CDN regardless |
| Deployment source | **Other** |

**Choose "Other", not GitHub.** The GitHub option commits a workflow of its own and wires a
credential into the repository. This repository already has its workflow
(`.github/workflows/site.yml`), and the credential goes in GitHub's secret store in step 3,
never in a file.

Or from the CLI:

```powershell
az staticwebapp create --name smallmodel-peterparker-ca --resource-group <resource-group> `
  --location eastus2 --sku Free --subscription <subscription>
```

A Static Web App serves one set of files to every hostname on it, so this is its own app, not
a second hostname on another site's app.

## 2. The DNS record, at Cloudflare

1. Copy the app's URL from its **Overview** page: `https://<generated-name>.azurestaticapps.net`.
2. In [the Cloudflare dashboard](https://dash.cloudflare.com), open `peterparker.ca`, then
   **DNS**, **Records**, **Add record**:

   | Type | Name | Target | Proxy status | TTL |
   |---|---|---|---|---|
   | CNAME | `smallmodel` | `<generated-name>.azurestaticapps.net` | **DNS only** | Auto |

3. **Proxy status DNS only, the grey cloud, not the orange one.** A proxied record hides the
   target behind Cloudflare's addresses, Azure's validation cannot see the CNAME, and the
   certificate is never issued. Projects 01 and 08 both hit this.
4. In the Static Web App: **Settings**, **Custom domains**, **+ Add**, **Custom domain on
   other DNS**. Enter `smallmodel.peterparker.ca`, record type **CNAME**, add
   ([Microsoft's page](https://learn.microsoft.com/en-us/azure/static-web-apps/custom-domain-external)).
   From the CLI instead:

   ```powershell
   az staticwebapp hostname set --name smallmodel-peterparker-ca --resource-group <resource-group> `
     --hostname smallmodel.peterparker.ca --subscription <subscription>
   ```

5. Wait for validation, usually minutes, occasionally an hour. Azure issues and renews the
   certificate itself.

## 3. The deploy token and the switch, in GitHub

The workflow uploads with the app's deployment token. It goes into the repository's secret
store straight from the Azure CLI, so it never appears on screen, in a file, in shell history
or in a chat. **The token alone is enough to publish to the site.**

```powershell
az staticwebapp secrets list --name smallmodel-peterparker-ca --resource-group <resource-group> `
  --subscription <subscription> --query properties.apiKey -o tsv |
  gh secret set AZURE_STATIC_WEB_APPS_API_TOKEN --repo Peter-A-P/fraction-of-the-bill

gh variable set SITE --body on --repo Peter-A-P/fraction-of-the-bill
```

Until `SITE` is `on`, every run of the website workflow skips rather than failing.

## 4. Publish

Automatic from here: every push to `main` that changes `site/` uploads it. To publish now
without a push:

```powershell
gh workflow run site.yml --repo Peter-A-P/fraction-of-the-bill
gh run watch --repo Peter-A-P/fraction-of-the-bill
```

Then add the address to the project's entry in peterparker.ca's `projects.yaml` as its
`demo:`, with a `demo_note:`, so the project page carries the button.

## 5. Check it

- Before trusting a change, look at it locally with the host's headers:
  `smallprint site-preview`, and open <http://localhost:8080>. **Not
  `python -m http.server`**, which sends none of the headers in `staticwebapp.config.json`
  and so shows a page the content security policy would partly refuse (project 08 lost a
  chart on its live site for two weeks that way).
- `https://smallmodel.peterparker.ca` serves over HTTPS with no certificate warning.
- The browser console on the live page is empty: a policy violation is reported there and
  nowhere else.
- The hero's figures match the README's: the 2B's cost per 1,000 on its cheapest card, and
  its break-even against luna and against sol.

Cost: the Free plan, CA$0.
