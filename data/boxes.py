from binsolver import BinSolver, PackRequest, Item, Bin

client = BinSolver(api_key="bs_debf5a55fd7f40caa45823d9f2563eb13f1749e64ebb4e3980e666b2a5deffbb")

request = PackRequest(
    objective="minBins",
    allowUnplaced=True,
    items=[
        Item(
            id="small-box",
            w=19,
            h=6,
            d=13,
            quantity=1000
        )
    ],
    bins=[
        Bin(
            id="main-container",
            w=40,
            h=30,
            d=35,
            quantity=1
        )
    ]
)

# 🔧 FORCE proper JSON serialization
payload = request.model_dump(mode="json")

response = client._client.post("/v1/pack", json=payload)
response.raise_for_status()

data = response.json()

print("Placed:", data["stats"]["placed"])
print("Unplaced:", data["stats"]["unplaced"])