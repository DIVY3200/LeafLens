"""Human-readable information for every class a LeafLens model can predict.

Keys are class labels in PlantVillage's "Plant___Condition" style. The 38
PlantVillage classes come first; the FIELDPLANT section adds the conditions that
only the FieldPlant dataset covers (see plant_disease/datasets.py for the mapping).

``PLANTVILLAGE_CLASSES`` is the output order of the original CNN
(trained_plant_disease_model.keras). Newer models store their own class list in a
JSON "model card" next to the .keras file.
"""


def _healthy(plant, tips):
    return dict(
        plant=plant,
        name="Healthy",
        type=None,
        cause=None,
        summary=f"No signs of disease were found on this {plant.lower()} leaf. Keep an eye on it as the season changes.",
        symptoms=[],
        treatment=[],
        prevention=tips,
    )


KNOWLEDGE = {
    # ---------------------------------------------------------------- Apple
    "Apple___Apple_scab": dict(
        plant="Apple", name="Apple scab", type="fungus",
        cause="Venturia inaequalis",
        summary="The most common apple disease worldwide. It thrives in cool, wet spring weather and can defoliate trees and leave fruit cracked and unsellable.",
        symptoms=[
            "Olive-green, velvety spots on leaves that turn dark brown to black",
            "Rough, corky, scab-like lesions on fruit",
            "Yellowing leaves and early leaf drop in bad years",
        ],
        treatment=[
            "Spray a fungicide such as captan, myclobutanil or sulfur from green tip through petal fall, every 7–10 days in wet weather",
            "Pick off and destroy badly infected leaves and fruit",
        ],
        prevention=[
            "Rake and destroy fallen leaves in autumn; the fungus overwinters in them",
            "Plant scab-resistant varieties such as Liberty, Enterprise or Freedom",
            "Prune to open the canopy so leaves dry quickly after rain",
        ],
    ),
    "Apple___Black_rot": dict(
        plant="Apple", name="Black rot", type="fungus",
        cause="Botryosphaeria obtusa",
        summary="A fungal disease that attacks leaves, fruit and bark. On leaves it is often called 'frog-eye leaf spot'.",
        symptoms=[
            "Small purple spots that grow into brown circles with a purple edge and tan centre",
            "Fruit rots from the blossom end with dark concentric rings, then shrivels into a black mummy",
            "Sunken, reddish-brown cankers on branches",
        ],
        treatment=[
            "Prune out cankers and dead wood at least 15 cm below visible damage",
            "Remove all mummified fruit from the tree and the ground",
            "Apply captan or another labelled fungicide from bloom through summer",
        ],
        prevention=[
            "Keep trees vigorous and avoid wounds; the fungus enters through damaged bark",
            "Clear prunings, dead wood and fallen fruit from the orchard",
        ],
    ),
    "Apple___Cedar_apple_rust": dict(
        plant="Apple", name="Cedar apple rust", type="fungus",
        cause="Gymnosporangium juniperi-virginianae",
        summary="A rust fungus that needs both apple and juniper (eastern red cedar) trees to complete its life cycle. Spores blow from juniper galls to apple leaves in spring.",
        symptoms=[
            "Bright yellow-orange spots on the upper side of leaves",
            "Tiny black dots in the centre of the spots, and tube-like growths on the underside",
            "Orange lesions on fruit and early leaf drop",
        ],
        treatment=[
            "Existing spots cannot be cured; protect new growth with myclobutanil or mancozeb from pink bud until a few weeks after petal fall",
            "Remove heavily infected leaves to improve airflow",
        ],
        prevention=[
            "Remove nearby junipers or cut out their orange galls in late winter where practical",
            "Grow rust-resistant varieties such as Liberty, Freedom or Redfree",
        ],
    ),
    "Apple___healthy": _healthy("Apple", [
        "Check leaves weekly during wet spring weather, when scab and rust infections start",
        "Rake up fallen leaves in autumn to remove overwintering disease",
        "Prune in late winter to let light and air into the canopy",
    ]),

    # ------------------------------------------------------------ Blueberry
    "Blueberry___healthy": _healthy("Blueberry", [
        "Keep the soil acidic (pH 4.5–5.5); yellowing between veins often means the pH is too high",
        "Mulch with pine bark or wood chips to keep shallow roots cool and moist",
        "Water at the base during dry spells, as blueberry roots dry out quickly",
    ]),

    # --------------------------------------------------------------- Cherry
    "Cherry_(including_sour)___Powdery_mildew": dict(
        plant="Cherry", name="Powdery mildew", type="fungus",
        cause="Podosphaera clandestina",
        summary="A fungus that coats young leaves and shoots in white powder. Unlike most fungi it needs humidity, not rain, so it spreads in warm, dry, humid weather.",
        symptoms=[
            "White, powdery patches on leaves, often starting on the underside",
            "Young leaves curl upward, pucker or look distorted",
            "White fungal growth on fruit stems and fruit near harvest",
        ],
        treatment=[
            "Spray sulfur, potassium bicarbonate or a labelled fungicide at the first sign of powder",
            "Prune out and destroy heavily infected shoots",
        ],
        prevention=[
            "Prune for an open canopy and good airflow",
            "Avoid excess nitrogen, which produces soft, susceptible growth",
            "Control root suckers, which are often infected first",
        ],
    ),
    "Cherry_(including_sour)___healthy": _healthy("Cherry", [
        "Prune after harvest to keep the centre of the tree open",
        "Watch new shoots for white powdery patches in warm, humid weather",
        "Clear fallen leaves and fruit to reduce leaf spot and brown rot",
    ]),

    # ----------------------------------------------------------------- Corn
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": dict(
        plant="Corn (maize)", name="Gray leaf spot", type="fungus",
        cause="Cercospora zeae-maydis",
        summary="A major yield-robbing leaf disease of corn that builds up in fields where corn residue is left on the surface and summers are warm and humid.",
        symptoms=[
            "Narrow, rectangular tan to gray lesions that run parallel to the leaf veins",
            "Lesions start on the lower leaves and move upward",
            "Whole leaves can turn gray and die in severe cases",
        ],
        treatment=[
            "If lesions reach the upper leaves around tasselling, a strobilurin or triazole fungicide can protect yield",
        ],
        prevention=[
            "Plant hybrids with good gray leaf spot resistance",
            "Rotate to a non-grass crop for at least one year",
            "Bury or break down infected residue with tillage",
        ],
    ),
    "Corn_(maize)___Common_rust_": dict(
        plant="Corn (maize)", name="Common rust", type="fungus",
        cause="Puccinia sorghi",
        summary="A rust fungus whose spores are carried long distances by wind. It is usually mild on field corn but can hurt sweet corn and susceptible hybrids.",
        symptoms=[
            "Small, oval, cinnamon-brown pustules on both sides of the leaf",
            "Pustules break open and release rusty, powdery spores that rub off",
            "Pustules turn dark brown to black late in the season",
        ],
        treatment=[
            "Usually no treatment is needed on field corn",
            "On sweet corn or susceptible hybrids, apply a fungicide if pustules appear early and spread quickly",
        ],
        prevention=[
            "Choose rust-resistant hybrids",
            "Plant early so the crop matures before rust peaks",
        ],
    ),
    "Corn_(maize)___Northern_Leaf_Blight": dict(
        plant="Corn (maize)", name="Northern leaf blight", type="fungus",
        cause="Exserohilum turcicum",
        summary="A leaf blight favoured by moderate temperatures and long periods of leaf wetness. Early infections can cut yield significantly.",
        symptoms=[
            "Long (2.5–15 cm), cigar-shaped gray-green to tan lesions",
            "Lesions start on lower leaves and spread upward",
            "Dark spore masses inside lesions in humid weather",
        ],
        treatment=[
            "Apply a fungicide at tasselling to silking if lesions are present on the upper leaves and the weather stays wet",
        ],
        prevention=[
            "Plant resistant hybrids",
            "Rotate crops and manage infected residue",
        ],
    ),
    "Corn_(maize)___healthy": _healthy("Corn (maize)", [
        "Rotate with a non-grass crop to break leaf-blight and leaf-spot cycles",
        "Scout the lower leaves from knee height onward; most leaf diseases start there",
        "A yellow 'V' down the midrib of lower leaves signals nitrogen deficiency",
    ]),

    # ---------------------------------------------------------------- Grape
    "Grape___Black_rot": dict(
        plant="Grape", name="Black rot", type="fungus",
        cause="Guignardia bidwellii",
        summary="One of the most destructive grape diseases in warm, humid climates. It can wipe out a crop by turning berries into hard black mummies.",
        symptoms=[
            "Small, round, reddish-brown leaf spots with a dark border",
            "Tiny black dots (spore bodies) inside the spots",
            "Berries turn brown, then shrivel into hard, black mummies",
        ],
        treatment=[
            "Spray mancozeb, myclobutanil or captan from early shoot growth until about four weeks after bloom",
            "Remove infected leaves and clusters as soon as they appear",
        ],
        prevention=[
            "Remove every mummified berry and infected cane during winter pruning",
            "Open the canopy so leaves and clusters dry quickly",
        ],
    ),
    "Grape___Esca_(Black_Measles)": dict(
        plant="Grape", name="Esca (black measles)", type="fungus",
        cause="A complex of wood-decaying fungi, including Phaeomoniella chlamydospora",
        summary="A trunk disease in which fungi slowly decay the vine's wood. Leaf symptoms come and go from year to year, but the infection persists.",
        symptoms=[
            "'Tiger stripe' leaves: yellow or red bands between the veins that turn brown",
            "Small dark spots on berries, which may crack",
            "Sudden wilting and collapse of a whole vine in hot weather",
        ],
        treatment=[
            "There is no cure; cut out dead wood and retrain a new trunk from a healthy shoot",
            "Remove and destroy badly affected vines",
        ],
        prevention=[
            "Prune in dry weather and protect large pruning cuts with a wound sealant",
            "Plant certified, disease-free nursery stock",
        ],
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": dict(
        plant="Grape", name="Leaf blight (Isariopsis leaf spot)", type="fungus",
        cause="Pseudocercospora vitis",
        summary="A late-season leaf spot that is most common in warm, humid regions. Heavy infections cause early leaf drop and weaken the vine.",
        symptoms=[
            "Irregular dark red-brown to black spots on leaves",
            "Spots often have a yellow edge and merge into large dead patches",
            "Early leaf drop late in the season",
        ],
        treatment=[
            "Remove infected leaves",
            "The mancozeb or copper sprays used for other grape diseases also control this one",
        ],
        prevention=[
            "Clear fallen leaves from under the vines",
            "Manage the canopy for good airflow",
        ],
    ),
    "Grape___healthy": _healthy("Grape", [
        "Thin shoots and remove leaves around clusters so the canopy dries quickly",
        "Start protective sprays at early shoot growth if black rot or mildew appeared last year",
        "Remove mummified berries and dead wood during winter pruning",
    ]),

    # --------------------------------------------------------------- Orange
    "Orange___Haunglongbing_(Citrus_greening)": dict(
        plant="Orange", name="Huanglongbing (citrus greening)", type="bacterium",
        cause="Candidatus Liberibacter asiaticus, spread by the Asian citrus psyllid",
        summary="The most serious citrus disease in the world. The bacteria block the tree's sap flow and there is no cure, so infected trees decline and eventually die.",
        symptoms=[
            "Blotchy, uneven yellowing that crosses the leaf veins (unlike nutrient deficiency, which follows them)",
            "Yellow shoots and thin canopy with twig dieback",
            "Small, lopsided, bitter fruit that stays green at the bottom",
        ],
        treatment=[
            "There is no cure; remove infected trees to protect healthy ones",
            "Report suspected cases to your local agriculture department, since it is a regulated disease in many regions",
        ],
        prevention=[
            "Control the Asian citrus psyllid that spreads the disease",
            "Buy only certified, disease-free nursery trees",
            "Inspect new flush growth regularly for psyllids",
        ],
    ),

    # ---------------------------------------------------------------- Peach
    "Peach___Bacterial_spot": dict(
        plant="Peach", name="Bacterial spot", type="bacterium",
        cause="Xanthomonas arboricola pv. pruni",
        summary="A bacterial disease spread by wind-driven rain. It is worst on sandy soils and in windy locations, and it damages both leaves and fruit.",
        symptoms=[
            "Small, angular, purple-black spots on leaves",
            "Spot centres fall out, leaving a 'shot-hole' look",
            "Yellow leaves that drop early; pitted, cracked spots on fruit",
        ],
        treatment=[
            "Apply copper sprays in the dormant season and at leaf fall",
            "Use only low-rate, labelled bactericides during the season, since copper can burn peach leaves",
        ],
        prevention=[
            "Plant resistant varieties",
            "Avoid excess nitrogen, and plant windbreaks on exposed sites",
        ],
    ),
    "Peach___healthy": _healthy("Peach", [
        "Apply a dormant spray in late winter if leaf curl or bacterial spot has been a problem",
        "Thin fruit so they don't touch; this improves size and reduces rot",
        "Prune to an open-centre shape for sunlight and airflow",
    ]),

    # --------------------------------------------------------- Bell pepper
    "Pepper,_bell___Bacterial_spot": dict(
        plant="Bell pepper", name="Bacterial spot", type="bacterium",
        cause="Xanthomonas species",
        summary="A bacterial disease that spreads in warm, wet weather through splashing water, tools and hands. It often arrives on infected seed or transplants.",
        symptoms=[
            "Small, water-soaked spots on leaves that turn brown, often with a yellow halo",
            "Yellowing and heavy leaf drop, which exposes fruit to sunscald",
            "Raised, scabby spots on fruit",
        ],
        treatment=[
            "Spray a copper-based bactericide (often mixed with mancozeb) to slow the spread",
            "Remove and destroy badly infected plants",
        ],
        prevention=[
            "Use certified disease-free seed and transplants, and resistant varieties",
            "Water at the base and avoid handling plants while they are wet",
            "Rotate away from peppers and tomatoes for 2–3 years",
        ],
    ),
    "Pepper,_bell___healthy": _healthy("Bell pepper", [
        "Water deeply and evenly; irregular watering causes blossom-end rot",
        "Mulch to keep soil from splashing onto leaves",
        "Rotate away from tomato, potato and eggplant for 2–3 years",
    ]),

    # --------------------------------------------------------------- Potato
    "Potato___Early_blight": dict(
        plant="Potato", name="Early blight", type="fungus",
        cause="Alternaria solani",
        summary="A common fungal leaf disease that hits older, stressed plants first. It reduces yield by killing leaves before the tubers have fully grown.",
        symptoms=[
            "Dark brown spots with concentric rings, like a target or bullseye",
            "Spots often have a yellow halo and start on the oldest, lowest leaves",
            "Dark, slightly sunken patches on tubers",
        ],
        treatment=[
            "Remove infected lower leaves",
            "Spray chlorothalonil, mancozeb or azoxystrobin at the first signs",
        ],
        prevention=[
            "Rotate away from potatoes, tomatoes and other nightshades for 2–3 years",
            "Keep plants well fed and watered; stressed plants are more susceptible",
            "Destroy infected plant debris after harvest",
        ],
    ),
    "Potato___Late_blight": dict(
        plant="Potato", name="Late blight", type="oomycete",
        cause="Phytophthora infestans",
        summary="The disease behind the Irish potato famine. In cool, wet weather it can destroy a whole field within a week, so act immediately.",
        symptoms=[
            "Large, pale green, water-soaked patches that quickly turn brown or black",
            "White, fuzzy growth on the underside of leaves in humid conditions",
            "Reddish-brown, granular rot inside tubers",
        ],
        treatment=[
            "Act fast: pull out infected plants, bag them and bin them (do not compost)",
            "Protect nearby plants with chlorothalonil, mancozeb or a blight-specific fungicide",
            "Warn neighbouring growers, since spores travel on the wind",
        ],
        prevention=[
            "Plant certified seed potatoes and destroy volunteer plants and cull piles",
            "Hill soil over tubers to keep spores from reaching them",
            "Grow resistant varieties and watch local blight forecasts",
        ],
    ),
    "Potato___healthy": _healthy("Potato", [
        "Hill soil around stems so tubers stay covered and protected",
        "Watch for dark, water-soaked patches in cool, wet weather; late blight moves fast",
        "Plant certified seed potatoes and rotate for 3 years away from other nightshades",
    ]),

    # ------------------------------------------------------------ Raspberry
    "Raspberry___healthy": _healthy("Raspberry", [
        "Remove canes that fruited this year right after harvest",
        "Keep rows narrow (30–45 cm) so canes dry quickly after rain",
        "Use drip irrigation to keep foliage dry",
    ]),

    # -------------------------------------------------------------- Soybean
    "Soybean___healthy": _healthy("Soybean", [
        "Rotate with corn or small grains to reduce residue-borne diseases",
        "Scout weekly from flowering onward and watch the upper leaves for spots",
        "Choose varieties rated resistant to the diseases common in your area",
    ]),

    # --------------------------------------------------------------- Squash
    "Squash___Powdery_mildew": dict(
        plant="Squash", name="Powdery mildew", type="fungus",
        cause="Podosphaera xanthii and related fungi",
        summary="The most common problem on squash, pumpkins and cucumbers in mid to late summer. It weakens plants and reduces fruit size and flavour.",
        symptoms=[
            "White, powdery spots on both sides of the leaves",
            "Spots spread until whole leaves look dusted with flour",
            "Infected leaves turn yellow, then brown and brittle",
        ],
        treatment=[
            "Spray sulfur, potassium bicarbonate, neem oil or horticultural oil at the first spots",
            "Remove the worst-affected leaves",
        ],
        prevention=[
            "Grow mildew-resistant varieties",
            "Plant in full sun with enough space for air to circulate",
        ],
    ),

    # ----------------------------------------------------------- Strawberry
    "Strawberry___Leaf_scorch": dict(
        plant="Strawberry", name="Leaf scorch", type="fungus",
        cause="Diplocarpon earlianum",
        summary="A fungal leaf disease that weakens plants and reduces the next season's yield. It spreads by splashing water.",
        symptoms=[
            "Many small, irregular purple spots on the upper leaf surface",
            "Spots merge until the leaf looks burned or scorched",
            "Leaf edges dry out and curl upward",
        ],
        treatment=[
            "Remove and destroy infected leaves",
            "Apply captan or another labelled fungicide in early spring and after renovation",
        ],
        prevention=[
            "Renovate beds after harvest by mowing old leaves and thinning plants",
            "Use drip irrigation and give plants enough space",
            "Plant resistant varieties",
        ],
    ),
    "Strawberry___healthy": _healthy("Strawberry", [
        "Renovate beds after harvest by mowing old leaves and thinning crowns",
        "Use straw mulch to keep fruit and leaves off the soil",
        "Water in the morning at soil level so foliage dries during the day",
    ]),

    # --------------------------------------------------------------- Tomato
    "Tomato___Bacterial_spot": dict(
        plant="Tomato", name="Bacterial spot", type="bacterium",
        cause="Xanthomonas species",
        summary="A bacterial disease that thrives in warm, wet weather and spreads by splashing water, tools and hands. It is often introduced on infected seed or transplants.",
        symptoms=[
            "Small (under 3 mm), dark, greasy-looking spots on leaves, sometimes with a yellow halo",
            "Yellowing leaves that drop early",
            "Raised, scabby spots on green fruit",
        ],
        treatment=[
            "Spray a copper bactericide (often mixed with mancozeb) to protect new growth",
            "Remove infected leaves and badly affected plants",
        ],
        prevention=[
            "Use certified disease-free seed and transplants",
            "Water at the base and disinfect tools",
            "Rotate away from tomatoes and peppers for 2–3 years",
        ],
    ),
    "Tomato___Early_blight": dict(
        plant="Tomato", name="Early blight", type="fungus",
        cause="Alternaria solani",
        summary="One of the most common tomato diseases. It starts on the oldest leaves and moves up the plant, reducing yield and exposing fruit to sunscald.",
        symptoms=[
            "Brown spots with concentric rings like a target, often with a yellow halo",
            "Starts on the lowest, oldest leaves",
            "Dark, sunken lesions on stems and near the stem end of fruit",
        ],
        treatment=[
            "Remove infected lower leaves and destroy them",
            "Spray chlorothalonil, copper or mancozeb at the first signs",
        ],
        prevention=[
            "Mulch to stop soil splashing spores onto leaves",
            "Stake and prune plants for airflow, and water at the base",
            "Rotate away from nightshades for 2–3 years",
        ],
    ),
    "Tomato___Late_blight": dict(
        plant="Tomato", name="Late blight", type="oomycete",
        cause="Phytophthora infestans",
        summary="A fast, destructive disease of tomatoes and potatoes. In cool, wet weather it can kill plants within days, so act immediately.",
        symptoms=[
            "Large, gray-green, water-soaked patches that quickly turn brown",
            "White, fuzzy growth on the underside of leaves in humid weather",
            "Firm, greasy-looking brown patches on fruit",
        ],
        treatment=[
            "Act fast: pull out infected plants, bag them and bin them (do not compost)",
            "Protect remaining plants with chlorothalonil, mancozeb or a blight-specific fungicide",
        ],
        prevention=[
            "Plant resistant varieties and keep tomatoes away from potatoes",
            "Water at the base and keep foliage dry",
            "Remove volunteer tomato and potato plants",
        ],
    ),
    "Tomato___Leaf_Mold": dict(
        plant="Tomato", name="Leaf mold", type="fungus",
        cause="Passalora fulva (Cladosporium fulvum)",
        summary="A fungal disease of high-humidity environments, most common in greenhouses and polytunnels where humidity stays above 85%.",
        symptoms=[
            "Pale green to yellow spots on the upper side of leaves",
            "Olive-green to grayish-purple velvety mould on the underside",
            "Leaves curl, wither and drop",
        ],
        treatment=[
            "Lower humidity right away by ventilating and spacing plants",
            "Remove infected leaves, and spray chlorothalonil or copper if needed",
        ],
        prevention=[
            "Grow leaf-mould-resistant varieties",
            "Keep greenhouse humidity below 85% and avoid wetting leaves",
        ],
    ),
    "Tomato___Septoria_leaf_spot": dict(
        plant="Tomato", name="Septoria leaf spot", type="fungus",
        cause="Septoria lycopersici",
        summary="A very common leaf disease that can strip a plant of its lower leaves, weakening it and exposing fruit to sunscald.",
        symptoms=[
            "Many small (1.5–6 mm) round spots with dark brown edges and gray or tan centres",
            "Tiny black specks in the centre of each spot",
            "Starts on the lowest leaves, which then yellow and drop",
        ],
        treatment=[
            "Remove infected lower leaves as soon as spots appear",
            "Spray chlorothalonil, mancozeb or copper to protect new leaves",
        ],
        prevention=[
            "Mulch, water at the base and rotate crops",
            "Remove plant debris and nightshade weeds that host the fungus",
        ],
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": dict(
        plant="Tomato", name="Two-spotted spider mite", type="pest",
        cause="Tetranychus urticae",
        summary="Not a disease but a tiny sap-sucking mite. Populations explode in hot, dry weather and can kill leaves within a couple of weeks.",
        symptoms=[
            "Fine yellow or white speckling (stippling) on the leaves",
            "Fine webbing on the underside of leaves and between stems",
            "Leaves turn bronze, dry out and fall",
        ],
        treatment=[
            "Blast the undersides of leaves with a strong spray of water",
            "Apply insecticidal soap, horticultural oil or neem oil, covering the leaf undersides",
            "Release predatory mites (Phytoseiulus persimilis) for biological control",
        ],
        prevention=[
            "Keep plants well watered; drought-stressed plants attract mites",
            "Avoid broad-spectrum insecticides that kill the mites' natural predators",
        ],
    ),
    "Tomato___Target_Spot": dict(
        plant="Tomato", name="Target spot", type="fungus",
        cause="Corynespora cassiicola",
        summary="A fungal disease of warm, humid climates that attacks leaves, stems and fruit and can cause heavy leaf loss.",
        symptoms=[
            "Small, water-soaked spots that grow into brown lesions with concentric rings",
            "Lesions have light brown centres and often a yellow halo",
            "Sunken, pitted spots on fruit",
        ],
        treatment=[
            "Remove infected lower leaves",
            "Spray chlorothalonil, mancozeb or azoxystrobin",
        ],
        prevention=[
            "Prune and stake plants for airflow, and avoid overhead watering",
            "Rotate crops and remove plant debris after harvest",
        ],
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": dict(
        plant="Tomato", name="Yellow leaf curl virus", type="virus",
        cause="Tomato yellow leaf curl virus (TYLCV), spread by whiteflies",
        summary="A devastating virus spread by the silverleaf whitefly. Plants infected young may produce almost no fruit.",
        symptoms=[
            "Leaves curl upward and cup inward",
            "Yellow leaf edges and between the veins",
            "Stunted growth, small leaves and flowers that drop",
        ],
        treatment=[
            "There is no cure; pull out infected plants and bag them to stop the spread",
            "Control whiteflies with yellow sticky traps and insecticidal soap",
        ],
        prevention=[
            "Plant TYLCV-resistant varieties",
            "Cover seedlings with insect-proof netting and control nearby weeds",
            "Use reflective mulch to repel whiteflies",
        ],
    ),
    "Tomato___Tomato_mosaic_virus": dict(
        plant="Tomato", name="Tomato mosaic virus", type="virus",
        cause="Tomato mosaic virus (ToMV)",
        summary="A very stable virus spread by hands, tools and seed. It can survive for months on surfaces and in plant debris.",
        symptoms=[
            "Light and dark green mottled, mosaic pattern on leaves",
            "Leaves look twisted or fern-like, and plants are stunted",
            "Fruit ripens unevenly, with yellow blotches",
        ],
        treatment=[
            "There is no cure; remove and destroy infected plants",
            "Disinfect tools, stakes and hands after handling infected plants",
        ],
        prevention=[
            "Grow resistant varieties and use certified seed",
            "Wash hands before working with plants, especially after handling tobacco",
        ],
    ),
    "Tomato___healthy": _healthy("Tomato", [
        "Stake or cage plants and remove lower leaves that touch the soil",
        "Water at the base, ideally in the morning, to keep foliage dry",
        "Mulch around plants to stop soil splashing spores onto leaves",
    ]),
}

# Output order of the original CNN: sorted(os.listdir("train")), which is what Keras used.
PLANTVILLAGE_CLASSES = sorted(KNOWLEDGE)

FIELDPLANT = {
    # -------------------------------------------------------------- Cassava
    "Cassava___Bacterial_blight": dict(
        plant="Cassava", name="Bacterial blight", type="bacterium",
        cause="Xanthomonas phaseoli pv. manihotis",
        summary="One of the most damaging cassava diseases. It spreads through infected stem cuttings and rain splash, and can kill whole shoots in wet seasons.",
        symptoms=[
            "Angular, water-soaked spots on leaves that turn brown",
            "Leaves wilt and dry from the edges, then drop",
            "Gummy ooze on stems and dieback of the shoot tips",
        ],
        treatment=[
            "No spray controls it; cut out and burn infected shoots, or uproot badly affected plants",
            "Disinfect cutting tools between plants",
        ],
        prevention=[
            "Plant cuttings only from healthy, disease-free mother plants",
            "Grow tolerant varieties recommended for your area",
            "Rotate fields and avoid working among wet plants",
        ],
    ),
    "Cassava___Brown_leaf_spot": dict(
        plant="Cassava", name="Brown leaf spot", type="fungus",
        cause="Passalora henningsii (Cercosporidium henningsii)",
        summary="A common fungal leaf spot that mostly affects older leaves. It is usually minor, but heavy infections cause early leaf drop.",
        symptoms=[
            "Small, angular brown spots with a darker border on the upper leaf surface",
            "Grayish spore growth inside the spots on the underside",
            "Older leaves turn yellow and fall early",
        ],
        treatment=[
            "Remove heavily spotted lower leaves",
            "Fungicide sprays are rarely worth it; focus on sanitation and tolerant varieties",
        ],
        prevention=[
            "Plant tolerant varieties",
            "Space plants for airflow and clear fallen leaves",
        ],
    ),
    "Cassava___Mosaic_disease": dict(
        plant="Cassava", name="Cassava mosaic disease", type="virus",
        cause="Cassava mosaic begomoviruses, spread by whiteflies and infected cuttings",
        summary="The most important cassava disease in Africa. Infected plants produce far fewer and smaller roots, and there is no cure.",
        symptoms=[
            "Yellow or pale-green mosaic patches on the leaves",
            "Leaves twisted, puckered or smaller than normal",
            "Stunted plants with short stems",
        ],
        treatment=[
            "There is no cure; uproot and destroy infected plants early",
            "Never take cuttings from infected plants",
        ],
        prevention=[
            "Plant virus-free cuttings of mosaic-resistant varieties",
            "Control whiteflies and remove infected plants nearby",
        ],
    ),
    "Cassava___healthy": _healthy("Cassava", [
        "Take cuttings only from healthy, vigorous mother plants",
        "Check young leaves weekly for mosaic patterns and whiteflies",
        "Keep the field weeded for the first three months",
    ]),

    # ----------------------------------------------------------------- Corn
    "Corn_(maize)___Brown_spot": dict(
        plant="Corn (maize)", name="Brown spot", type="fungus",
        cause="Often Physoderma maydis; other leaf-spot fungi can look similar",
        summary="Physoderma brown spot appears in warm, wet weather when water collects in the leaf whorl. It rarely causes major yield loss.",
        symptoms=[
            "Small, round to oval, yellowish-brown or purplish-brown spots",
            "Spots often form bands across the leaf",
            "Dark purple-brown spots on the midrib and leaf sheath",
        ],
        treatment=[
            "Usually no treatment is needed",
            "A fungicide only pays off on high-value crops with heavy infection",
        ],
        prevention=[
            "Rotate crops and bury infected residue",
            "Grow tolerant hybrids",
        ],
    ),
    "Corn_(maize)___Downy_mildew": dict(
        plant="Corn (maize)", name="Downy mildew", type="oomycete",
        cause="Peronosclerospora species",
        summary="A serious disease in tropical Africa and Asia. Plants infected young often stay stunted and produce no cobs.",
        symptoms=[
            "Pale yellow or white streaks running along the leaves",
            "White, downy growth on the leaf underside on cool, humid mornings",
            "Stunted plants with malformed tassels or cobs",
        ],
        treatment=[
            "Infected plants cannot be cured; pull them out and destroy them early",
        ],
        prevention=[
            "Sow seed treated with metalaxyl (mefenoxam)",
            "Plant resistant varieties",
            "Rotate away from maize and sorghum, which share the disease",
        ],
    ),
    "Corn_(maize)___Streak_virus": dict(
        plant="Corn (maize)", name="Maize streak virus", type="virus",
        cause="Maize streak virus (MSV), spread by Cicadulina leafhoppers",
        summary="A major maize virus in Africa. Plants infected in their first weeks can lose most of their yield.",
        symptoms=[
            "Narrow, broken yellow streaks along the leaf veins",
            "Streaks start at the base of young leaves",
            "Stunted plants with small, poorly filled cobs",
        ],
        treatment=[
            "There is no cure; pull out plants infected early to slow the spread",
        ],
        prevention=[
            "Plant streak-resistant varieties",
            "Sow early in the rains, before leafhopper numbers build up",
            "Control the grassy weeds that host the leafhoppers",
        ],
    ),
    "Corn_(maize)___Stripe": dict(
        plant="Corn (maize)", name="Leaf stripe", type="virus",
        cause="Often maize stripe virus, spread by the corn planthopper",
        summary="Yellow stripes along corn leaves. Maize stripe virus is a common cause, but zinc or magnesium shortage can look similar.",
        symptoms=[
            "Continuous yellow stripes running along the leaf",
            "Young leaves may turn almost completely yellow",
            "Stunted plants, sometimes with poor or no cobs",
        ],
        treatment=[
            "If a virus is likely, pull out badly affected plants",
            "If many plants are striped evenly, test the soil for zinc and magnesium",
        ],
        prevention=[
            "Control planthoppers and the weeds around the field",
            "Plant tolerant varieties",
        ],
    ),
    "Corn_(maize)___Yellowing": dict(
        plant="Corn (maize)", name="Yellowing", type="disorder",
        cause="Usually nitrogen shortage, waterlogging or drought stress",
        summary="Yellow corn leaves are a sign of stress rather than one disease. Nitrogen shortage is the most common cause.",
        symptoms=[
            "Lower leaves yellow first, often in a 'V' from the tip down the midrib (nitrogen)",
            "Whole plants turn pale after heavy rain or flooding",
            "Yellowing in streaks may point to a virus instead",
        ],
        treatment=[
            "Side-dress with nitrogen if the pattern matches",
            "Improve drainage in waterlogged spots",
        ],
        prevention=[
            "Test the soil and fertilise according to the results",
            "Avoid planting in poorly drained areas",
        ],
    ),
    "Corn_(maize)___Smut": dict(
        plant="Corn (maize)", name="Common smut", type="fungus",
        cause="Ustilago maydis",
        summary="A fungus that turns parts of the plant into swollen galls full of black spores. It enters through wounds from hail, insects or cultivation.",
        symptoms=[
            "Silvery-white swellings (galls) on ears, tassels, stems or leaves",
            "Galls burst open and release masses of black, powdery spores",
            "Small galls along the leaf midrib",
        ],
        treatment=[
            "Cut off and destroy galls before they burst",
        ],
        prevention=[
            "Avoid injuring plants during weeding",
            "Grow tolerant hybrids and balance nitrogen fertiliser",
        ],
    ),

    # --------------------------------------------------------------- Tomato
    "Tomato___Blight": dict(
        plant="Tomato", name="Blight (early or late)", type="fungus",
        cause="Early blight (Alternaria solani) or late blight (Phytophthora infestans)",
        summary="Field photos of blight don't always show whether it is early or late blight. Late blight spreads much faster, so check the signs below.",
        symptoms=[
            "Early blight: brown spots with target-like rings, starting on old leaves",
            "Late blight: large, greasy gray-green patches that spread within days",
            "Late blight: white fuzz on the leaf underside in humid weather",
        ],
        treatment=[
            "Remove and destroy infected leaves; bag late-blight plants instead of composting them",
            "Spray chlorothalonil or mancozeb to protect healthy foliage",
        ],
        prevention=[
            "Water at the base and keep foliage dry",
            "Stake plants for airflow and rotate crops",
        ],
    ),
    "Tomato___Brown_spots": dict(
        plant="Tomato", name="Brown leaf spots", type="fungus",
        cause="Several diseases: early blight, Septoria leaf spot, target spot or bacterial spot",
        summary="Brown spots on tomato leaves in the field can come from several diseases. Look closely at the spots to narrow it down.",
        symptoms=[
            "Spots with concentric rings point to early blight or target spot",
            "Many tiny spots with gray centres point to Septoria leaf spot",
            "Small, greasy, dark spots point to bacterial spot",
        ],
        treatment=[
            "Remove spotted lower leaves and destroy them",
            "Spray a protectant such as chlorothalonil, mancozeb or copper",
        ],
        prevention=[
            "Mulch, stake plants and water at the base",
            "Rotate away from tomatoes, potatoes and peppers for 2–3 years",
        ],
    ),
    "Tomato___Bacterial_wilt": dict(
        plant="Tomato", name="Bacterial wilt", type="bacterium",
        cause="Ralstonia solanacearum",
        summary="A soil-borne bacterium that blocks the plant's water vessels. Plants wilt suddenly while the leaves are still green.",
        symptoms=[
            "Sudden wilting of green leaves, first in the heat of the day",
            "Brown discoloration inside the lower stem",
            "Milky ooze streams from a cut stem placed in clear water",
        ],
        treatment=[
            "There is no cure; pull out wilted plants with their roots and destroy them",
        ],
        prevention=[
            "Rotate for several years with non-host crops such as maize or beans",
            "Use resistant varieties or grafted plants, and avoid waterlogged soil",
        ],
    ),
}
KNOWLEDGE.update(FIELDPLANT)


def describe(label):
    """Knowledge entry for a label, with a generic fallback for classes that have no curated guide."""
    if label in KNOWLEDGE:
        return KNOWLEDGE[label]
    plant, _, condition = label.partition("___")
    healthy = condition.lower() == "healthy"
    return dict(
        plant=plant.replace("_", " ").strip() or label,
        name="Healthy" if healthy else condition.replace("_", " ").strip().capitalize(),
        type=None if healthy else "unknown",
        cause=None,
        summary="LeafLens recognises this pattern but has no detailed guide for it yet. Ask a local agricultural extension officer to confirm it before treating.",
        symptoms=[],
        treatment=[],
        prevention=["Remove badly affected leaves and keep checking nearby plants"],
    )


def supported_plants(classes=PLANTVILLAGE_CLASSES):
    """Plants a model knows, each with its list of conditions (diseases first)."""
    plants = {}
    for label in classes:
        entry = describe(label)
        plants.setdefault(entry["plant"], []).append(
            {"name": entry["name"], "healthy": entry["type"] is None}
        )
    return [
        {"name": plant, "conditions": sorted(conditions, key=lambda c: c["healthy"])}
        for plant, conditions in sorted(plants.items())
    ]
