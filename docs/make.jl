using HSVIforOSPOSGs
using Documenter

DocMeta.setdocmeta!(HSVIforOSPOSGs, :DocTestSetup, :(using HSVIforOSPOSGs); recursive=true)

makedocs(;
    modules=[HSVIforOSPOSGs],
    authors="Jakub Brož <brozjak2@fel.cvut.cz>",
    repo="https://github.com/brozjak2/HSVIforOSPOSGs.jl/blob/{commit}{path}#{line}",
    sitename="HSVIforOSPOSGs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://brozjak2.github.io/HSVIforOSPOSGs.jl",
        edit_link="master",
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => "manual.md",
        "Game format" => "game_format.md",
        "API Reference" => "api.md"
    ],
)

deploydocs(;
    repo="github.com/brozjak2/HSVIforOSPOSGs.jl",
    devbranch="master",
)
