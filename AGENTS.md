<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->


## Project architecture
- Keep NETWORLD UI data in `src/data/mockData.ts` and expose it through `src/services/api.ts`, so a future FastAPI integration does not require page changes.
- Keep each major intelligence view as its own TanStack route, wrapped by the shared command-center shell in the root route.
