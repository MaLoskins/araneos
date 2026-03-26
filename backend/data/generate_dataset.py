"""Generate a synthetic social network dataset for testing the Araneos pipeline.

Each user has a fixed 'user_type' (enthusiast/critic/analyst) that determines
their posting behavior. This is a per-user label that a GNN can actually learn
from text features + graph structure.
"""
import csv
import random
import os

random.seed(42)

NUM_USERS = 150
USERS = [f"user_{i:03d}" for i in range(NUM_USERS)]
CHANNELS = ["tech", "science", "gaming", "politics", "music", "sports", "finance", "health", "travel", "food"]

# Three user types with distinct behaviors:
# - enthusiast: mostly positive, uses excited language
# - critic: mostly negative, uses critical language
# - analyst: mostly neutral, uses analytical language
USER_TYPES = ["enthusiast", "critic", "analyst"]

USER_PROFILES = {}
for i, u in enumerate(USERS):
    # Assign type in balanced thirds
    utype = USER_TYPES[i % 3]
    prefs = random.sample(CHANNELS, random.randint(2, 4))
    USER_PROFILES[u] = {"type": utype, "channels": prefs}

TEMPLATES = {
    "tech": {
        "enthusiast": [
            "Just deployed {thing} and performance improved by {pct}%, absolutely love it!",
            "The new {thing} update is fantastic, incredible improvement over the last version!",
            "{thing} is a game changer, our entire team is excited about what we can build now!",
            "Switched to {thing} and never looking back, the developer experience is outstanding!",
            "Amazing progress on {thing} this quarter, the community is doing incredible work!",
            "Just discovered {thing} and I'm blown away by how elegant the design is!",
            "Our {thing} migration was a huge success, everything runs beautifully now!",
            "Excited to share that our {thing} project won the hackathon, team did amazing work!",
        ],
        "critic": [
            "The {thing} update broke our entire CI pipeline again, getting tired of this instability",
            "Frustrated with {thing} performance issues, memory leaks are getting worse not better",
            "Another breaking change in {thing} with no migration guide, this is unacceptable",
            "{thing} documentation is misleading and outdated, wasted an entire day on wrong approach",
            "Security vulnerability in {thing} should have been caught months ago, concerning negligence",
            "The {thing} API design is fundamentally flawed, they need to go back to the drawing board",
            "Why does {thing} still have this bug after three years? Quality control is nonexistent",
            "Tried {thing} for a week and went back to the old solution, the hype is not justified",
        ],
        "analyst": [
            "Benchmarking {thing} against alternatives, initial results show mixed performance tradeoffs",
            "Reviewing the {thing} changelog, notable changes in error handling and memory management",
            "Comparing {thing} adoption rates across enterprise vs startup environments this quarter",
            "The {thing} architecture has interesting tradeoffs between consistency and availability",
            "Documenting our {thing} evaluation criteria for the technical review board next week",
            "Running controlled tests on {thing} throughput under various load conditions today",
            "Analyzing {thing} security audit results, several findings need further investigation",
            "Compiling a comparison matrix for {thing} and three competing solutions for our team",
        ],
        "things": ["Kubernetes", "Docker", "React", "TypeScript", "Rust", "Go", "PostgreSQL", "Redis",
                    "GraphQL", "WebAssembly", "Terraform", "AWS Lambda", "NextJS", "Svelte", "MongoDB"],
    },
    "science": {
        "enthusiast": [
            "Breakthrough results in {thing} research, this could change everything we know!",
            "Published our {thing} findings today, so proud of what the team accomplished!",
            "The new {thing} technique shows incredible promise, truly revolutionary approach!",
            "Got our {thing} grant approved with flying colors, reviewers were very enthusiastic!",
            "Attended a brilliant {thing} seminar, the speaker's insights were truly inspiring!",
            "Our {thing} experiment exceeded all expectations, the results are remarkable!",
            "The {thing} field is advancing at an amazing pace, what an exciting time to be involved!",
        ],
        "critic": [
            "Peer review rejected our {thing} paper with contradictory and unhelpful feedback again",
            "Failed to replicate the {thing} results from that high profile paper, very concerning trend",
            "The {thing} community has a serious reproducibility crisis that nobody wants to address",
            "{thing} funding keeps getting cut while useless projects get millions, priorities are wrong",
            "Found critical methodology flaws in several recent {thing} publications, embarrassing oversight",
            "The {thing} dataset everyone relies on has significant labeling errors nobody caught",
            "Another {thing} retraction, the rush to publish is destroying scientific credibility",
        ],
        "analyst": [
            "Reviewing the latest {thing} meta-analysis, sample sizes vary considerably across studies",
            "Preparing our {thing} submission, carefully documenting methodology for reproducibility",
            "Collecting additional {thing} data to strengthen statistical power of our analysis",
            "The {thing} results suggest a moderate effect size, further investigation warranted",
            "Comparing {thing} approaches across different research groups to identify best practices",
            "Documenting {thing} experimental protocols for standardization across our lab network",
            "Reading through {thing} literature review, identifying gaps for our next research phase",
        ],
        "things": ["protein folding", "CRISPR", "dark matter", "quantum computing", "neural network",
                    "climate modeling", "gene therapy", "gravitational wave", "mRNA vaccine", "fusion energy",
                    "exoplanet", "machine learning", "brain imaging", "antibiotic resistance", "microbiome"],
    },
    "gaming": {
        "enthusiast": [
            "The new {thing} update is incredible, the developers really outdid themselves this time!",
            "Just finished {thing} and it's easily my game of the year, what a masterpiece!",
            "{thing} multiplayer is the most fun I've had in years, stayed up all night playing!",
            "The {thing} soundtrack is absolutely phenomenal, gives me chills every single time!",
            "Modding community for {thing} is amazing, the creativity never ceases to impress me!",
            "Cannot stop playing {thing}, it's perfectly designed and endlessly entertaining!",
        ],
        "critic": [
            "{thing} servers are down again, fourth outage this month with no communication from devs",
            "The {thing} microtransactions are predatory garbage ruining what could be a decent game",
            "Unplayable lag in {thing} ranked matches, the netcode is embarrassingly broken still",
            "{thing} launched in an inexcusably buggy state, clearly rushed out the door unfinished",
            "The {thing} anti-cheat is useless, every other match has obvious cheaters ruining it",
            "Lost 20 hours of {thing} progress to a save bug they still haven't fixed, pathetic",
        ],
        "analyst": [
            "Comparing {thing} frame rates across different GPU configurations for optimization guide",
            "The {thing} meta has shifted significantly since the balance patch, analyzing win rates",
            "Watching {thing} tournament replays to study positioning strategies at the highest level",
            "Testing {thing} on various hardware setups to document minimum viable specifications",
            "The {thing} beta has potential but several mechanics need tuning before full release",
            "Documenting {thing} progression systems and their impact on player retention metrics",
        ],
        "things": ["Elden Ring", "Baldurs Gate 3", "Palworld", "Helldivers 2", "Cyberpunk 2077",
                    "Starfield", "Zelda TOTK", "FF16", "Diablo 4", "Street Fighter 6",
                    "Valorant", "Apex Legends", "Fortnite", "Minecraft", "Stardew Valley"],
    },
    "politics": {
        "enthusiast": [
            "The new {thing} legislation is a wonderful step forward, proud of our representatives!",
            "Impressed by the bipartisan {thing} effort, this gives me real hope for the future!",
            "Record voter turnout for {thing} shows democracy is alive and thriving, beautiful to see!",
            "The {thing} initiative is already making a positive difference in local communities!",
            "Great to see young people engaged in {thing} activism, the next generation inspires me!",
        ],
        "critic": [
            "The {thing} bill is deeply flawed and ignores the people it claims to help, typical",
            "Politicians keep making empty {thing} promises while nothing actually gets better",
            "Media coverage of {thing} is incredibly biased in both directions, impossible to trust",
            "The {thing} policy will hurt vulnerable populations most, as usual nobody considers them",
            "Corruption in the {thing} program is outrageous and nobody is being held accountable",
        ],
        "analyst": [
            "Analyzing {thing} polling data across demographics, interesting regional variations emerge",
            "The {thing} proposal has complex implications, reading the full text before commenting",
            "Comparing {thing} positions across candidates based on their voting records this session",
            "The {thing} debate raised valid points on both sides, the tradeoffs are genuinely difficult",
            "Examining historical {thing} precedents to contextualize the current policy discussion",
        ],
        "things": ["infrastructure", "education reform", "healthcare", "climate policy", "housing",
                    "immigration", "tax reform", "digital privacy", "minimum wage", "election security"],
    },
    "music": {
        "enthusiast": [
            "The new {thing} album is pure genius, every track is an absolute masterpiece!",
            "Saw {thing} live last night and it was the best concert of my entire life, incredible!",
            "{thing} new single is on repeat all day, the production quality is just next level!",
            "The {thing} collab is everything I hoped for and more, these artists are phenomenal!",
            "Learning {thing} songs on guitar and loving every minute of it, so much creative energy!",
        ],
        "critic": [
            "Disappointed by the new {thing} album, feels lazy and uninspired compared to earlier work",
            "{thing} concert was overpriced for terrible sound quality, felt like a total waste of money",
            "The {thing} streaming situation is a mess, corporate greed is killing artistic expression",
            "{thing} completely sold out, the new direction abandons everything that made them special",
            "Ticket scalpers destroyed the {thing} tour, genuine fans shut out while bots profit",
        ],
        "analyst": [
            "Listening to the {thing} discography chronologically, tracking the production evolution",
            "The {thing} documentary provides interesting context about their creative process changes",
            "Comparing {thing} mixing techniques across albums using spectral analysis tools today",
            "Reviewing {thing} album sales data alongside streaming numbers for industry report",
            "Analyzing harmonic complexity in {thing} compositions compared to genre contemporaries",
        ],
        "things": ["Radiohead", "Kendrick Lamar", "Taylor Swift", "Tyler the Creator", "Billie Eilish",
                    "Arctic Monkeys", "SZA", "Daft Punk", "Beyonce", "Frank Ocean", "Tame Impala"],
    },
    "sports": {
        "enthusiast": [
            "What an unbelievable {thing} performance, that was the best game I have ever seen!",
            "The {thing} comeback was legendary, this team has so much heart and determination!",
            "{thing} development this season is incredible, we are witnessing greatness unfold!",
            "Historic {thing} achievement, I feel so lucky to have watched this live, unforgettable!",
            "The {thing} chemistry is perfect right now, every player is contributing at their peak!",
        ],
        "critic": [
            "Terrible {thing} officiating completely ruined what should have been a great game",
            "The {thing} injuries prove management doesnt care about player health or longevity",
            "{thing} front office needs to be fired, years of incompetent decisions caught up to them",
            "Another pathetic {thing} loss, this supposed rebuild is nothing but empty excuses",
            "The {thing} trade was a franchise-destroying disaster, gave away our future for nothing",
        ],
        "analyst": [
            "Analyzing {thing} advanced stats, the expected goals model tells a different story",
            "The {thing} draft class has depth at guard position, comparing combine measurements",
            "Reviewing {thing} film from the last three games, tactical adjustments are measurable",
            "Running regression analysis on {thing} salary cap impact across different roster builds",
            "Projecting {thing} playoff scenarios based on remaining schedule strength metrics",
        ],
        "things": ["Lakers", "Manchester City", "Yankees", "Warriors", "Chiefs", "Barcelona",
                    "Liverpool", "Dodgers", "Celtics", "Eagles", "Real Madrid", "Bayern Munich"],
    },
    "finance": {
        "enthusiast": [
            "Strong {thing} earnings beat estimates across the board, incredible quarter for investors!",
            "The {thing} rally shows amazing renewed confidence, exciting times for the market!",
            "Fantastic returns from {thing} this year, our diversification strategy paid off beautifully!",
            "The {thing} IPO was a massive success, strong demand signals a healthy market ahead!",
            "Positive {thing} indicators point to sustained growth, very optimistic about the outlook!",
        ],
        "critic": [
            "The {thing} crash wiped out billions overnight, regulators were asleep at the wheel again",
            "{thing} inflation is destroying purchasing power, working families are being crushed",
            "Another {thing} fraud scandal with zero accountability, the system protects the wealthy",
            "The {thing} bubble is obvious to anyone paying attention, this will end very badly",
            "{thing} inequality keeps growing while policymakers pretend everything is fine",
        ],
        "analyst": [
            "Reviewing {thing} quarterly reports, revenue growth offset by margin compression",
            "The {thing} interest rate decision depends on next months employment data release",
            "Rebalancing {thing} allocations based on updated volatility and correlation models",
            "Modeling {thing} scenarios using Monte Carlo simulation across different rate paths",
            "Comparing {thing} sector valuations against historical averages and forward estimates",
        ],
        "things": ["tech sector", "cryptocurrency", "S&P 500", "bond market", "real estate",
                    "banking", "commodities", "AI stocks", "green energy", "emerging markets"],
    },
    "health": {
        "enthusiast": [
            "Amazing {thing} breakthrough in clinical trials, this could help millions of people!",
            "The {thing} prevention program results are wonderful, real lives being improved!",
            "So excited about the {thing} research progress, the future of medicine looks bright!",
            "Completed a {thing} wellness program and feeling better than I have in years!",
            "The {thing} community health campaign reached millions, what an inspiring achievement!",
        ],
        "critic": [
            "{thing} misinformation online is dangerous and platforms refuse to take responsibility",
            "Insurance still refuses to cover {thing} treatment while patients suffer needlessly",
            "{thing} wait times are inhumane, the system is failing the people who need it most",
            "The {thing} side effects were covered up, patients deserve honesty from providers",
            "Critical {thing} workforce shortage is a crisis that has been ignored for far too long",
        ],
        "analyst": [
            "Evaluating {thing} clinical trial methodology, the double-blind design is appropriate",
            "The updated {thing} guidelines reflect recent evidence, reviewing changes systematically",
            "Comparing {thing} treatment outcomes across different patient populations and protocols",
            "Analyzing {thing} cost-effectiveness data for the health economics policy review",
            "Documenting {thing} intervention adherence rates to identify barriers to compliance",
        ],
        "things": ["mental health", "vaccination", "sleep quality", "nutrition", "physical therapy",
                    "cancer screening", "telemedicine", "chronic pain", "heart disease", "diabetes"],
    },
    "travel": {
        "enthusiast": [
            "Just visited {thing} and it was absolutely magical, exceeded every expectation I had!",
            "The {thing} locals were incredibly warm and welcoming, the food was out of this world!",
            "Found a stunning hidden gem in {thing} that most tourists completely miss, paradise!",
            "The {thing} architecture is breathtaking, every single street corner is photo worthy!",
            "Best trip of my life to {thing}, already counting down until I can go back again!",
        ],
        "critic": [
            "Overtourism has ruined {thing}, the crowds make it nearly impossible to enjoy anything",
            "Got scammed in {thing} by a fake tour operator, tourist traps are everywhere now",
            "The {thing} airport was a nightmare of delays, lost luggage, and rude staff throughout",
            "{thing} is massively overpriced and overhyped, would not recommend to anyone honestly",
            "Food poisoning in {thing} destroyed our entire vacation, hygiene standards were awful",
        ],
        "analyst": [
            "Planning a {thing} itinerary, comparing costs across different seasons and travel styles",
            "Reviewing {thing} accommodation options across price points for the travel guide update",
            "The {thing} visa requirements changed recently, documenting the updated application process",
            "Comparing {thing} transportation infrastructure for accessibility study this quarter",
            "Analyzing {thing} tourism impact data for the sustainable travel research project",
        ],
        "things": ["Japan", "Iceland", "Portugal", "New Zealand", "Morocco", "Croatia", "Vietnam",
                    "Peru", "Norway", "Thailand", "Greece", "Costa Rica", "Scotland", "Bali"],
    },
    "food": {
        "enthusiast": [
            "Made {thing} from scratch and it turned out absolutely perfect, so proud of myself!",
            "This {thing} recipe is life changing, the flavors are incredible and so satisfying!",
            "Found the best {thing} restaurant in the city, every dish was an absolute delight!",
            "The {thing} farmers market was amazing this week, so much beautiful fresh produce!",
            "Learning {thing} techniques has transformed my cooking, every meal is an adventure now!",
        ],
        "critic": [
            "The {thing} restaurant was awful, overpriced bland food with pretentious atmosphere",
            "Tried making {thing} following a popular recipe and it was a complete inedible disaster",
            "Food delivery {thing} arrived cold and wrong, getting a refund was impossibly frustrating",
            "The quality of {thing} ingredients keeps declining while prices keep climbing endlessly",
            "{thing} portions have shrunk dramatically while they charge even more, insulting honestly",
        ],
        "analyst": [
            "Testing three different {thing} recipes side by side to determine optimal technique",
            "Comparing {thing} cooking methods across temperature and timing variables systematically",
            "The {thing} cookbook takes an interesting molecular gastronomy approach worth examining",
            "Documenting {thing} ingredient sourcing across local versus imported supply chains",
            "Analyzing {thing} nutritional profiles across different preparation methods for study",
        ],
        "things": ["sourdough bread", "ramen", "Thai curry", "pasta carbonara", "sushi",
                    "BBQ ribs", "French pastry", "Korean fried chicken", "tacos al pastor", "dim sum"],
    },
}

def generate_message(channel, user_type):
    templates = TEMPLATES[channel][user_type]
    things = TEMPLATES[channel]["things"]
    template = random.choice(templates)
    thing = random.choice(things)
    msg = template.replace("{thing}", thing)
    if "{pct}" in msg:
        msg = msg.replace("{pct}", str(random.randint(15, 85)))
    return msg

rows = []
for _ in range(2500):
    user = random.choice(USERS)
    profile = USER_PROFILES[user]
    channel = random.choice(profile["channels"])
    user_type = profile["type"]
    message = generate_message(channel, user_type)

    # Pick reply target - preferentially reply to same type (homophily)
    same_type = [u for u in USERS if u != user and USER_PROFILES[u]["type"] == user_type
                 and channel in USER_PROFILES[u]["channels"]]
    diff_type = [u for u in USERS if u != user and u not in same_type
                 and channel in USER_PROFILES[u]["channels"]]

    if same_type and random.random() < 0.65:  # 65% homophily
        replied_to = random.choice(same_type)
    elif diff_type:
        replied_to = random.choice(diff_type)
    elif same_type:
        replied_to = random.choice(same_type)
    else:
        replied_to = random.choice([u for u in USERS if u != user])

    rows.append([user, channel, message, user_type, replied_to])

random.shuffle(rows)

out_path = os.path.join(os.path.dirname(__file__), "test_social_graph.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["user_id", "channel", "message", "user_type", "replied_to_user"])
    writer.writerows(rows)

print(f"Generated {len(rows)} rows -> {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1024:.1f} KB")

# Verify label consistency
from collections import Counter
user_types = {}
for r in rows:
    uid = r[0]
    utype = r[3]
    if uid in user_types:
        assert user_types[uid] == utype, f"Inconsistent type for {uid}"
    user_types[uid] = utype

type_dist = Counter(user_types.values())
print(f"User type distribution: {dict(type_dist)}")
print(f"Each user has exactly ONE consistent label across all their posts")
