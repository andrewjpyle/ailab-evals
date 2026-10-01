"""Deterministically (re)build every fixture under fixtures/ from the hand-authored content below.

All entities are FICTIONAL (invented towns, shops, services). Labels are fixed BY CONSTRUCTION:
each item is written to be a member of a category whose label is defined before any model sees
it (see FIXTURES.md). Run:  python scripts/build_fixtures.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "fixtures"
rng = random.Random(20261001)


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------
# 1. substance: "does this page give a reader specific, usable information?"  yes / no
#    yes = >= 3 concrete, checkable specifics (numbers, hours, prices, steps, named parts).
#    no  = boilerplate, vague praise, or specific-SOUNDING copy with no actual data.
# ---------------------------------------------------------------------------------------

SUBSTANTIVE = [
    "Marrowfield Public Pool opens May 27 and closes Sept 3. Lap swim runs 6:00-8:30 a.m. weekdays; a day pass is $6 for adults and $3 for kids under 12. The deep end is 12 ft and requires a swim test.",
    "To reset the Kestrel K-200 thermostat, hold MODE and FAN together for 8 seconds until the display reads 'rS'. Release, then press UP twice to confirm. Settings return to 68°F heat / 74°F cool.",
    "Brindle Hollow Farm sells half-shares of its 20-week vegetable CSA for $340. Pickup is Thursdays 3-7 p.m. at the red barn on Quarry Road. Members get roughly 8-10 lb of produce per week in peak season.",
    "Route 14 buses leave Ostrand Station every 20 minutes from 5:40 a.m. to 11:20 p.m. The trip to Westgate Mall takes 26 minutes and costs $2.25, or $1.10 with a reduced-fare card.",
    "The Pell Street community garden has 42 raised beds (4x8 ft). Annual plot fee is $35; water is included. The waitlist opens January 15 and plots are assigned by lottery on March 1.",
    "Larkspur Dental accepts new patients Tuesday-Saturday. A cleaning and exam without insurance costs $145; X-rays add $60. Same-day emergency slots open at 7:30 a.m. by phone only.",
    "Our sourdough uses 1,000 g bread flour, 750 g water, 200 g starter, and 22 g salt. Bulk ferment 5 hours at 76°F with four sets of folds, shape, then cold-proof 14 hours before baking at 475°F.",
    "Fenwick Library's makerspace has two 3D printers (0.4 mm nozzle, PLA only), a laser cutter limited to 3 mm plywood, and a vinyl cutter. Sessions are 90 minutes and require a 20-minute safety orientation first.",
    "Granite Creek Trail is 7.4 miles round trip with 1,150 ft of elevation gain. The lot fills by 8 a.m. on weekends; parking is $5 cash. Dogs must be leashed, and the upper falls section closes Nov 1-Apr 15.",
    "Ashby Tire's winter changeover costs $48 for four mounted wheels or $96 for mount-and-balance. Storage of off-season tires is $60 per season. Appointments run 45 minutes on average.",
    "The Ostrand recycling center accepts #1, #2, and #5 plastics, flattened cardboard, and glass sorted by color. It does not take plastic bags or styrofoam. Hours: Wed 10-4, Sat 8-2.",
    "Hazel Point Marina charges $14 per foot per month for summer slips (May-Oct), with a 20-ft minimum. Electricity is metered at $0.18/kWh. The fuel dock sells non-ethanol gas and closes at 6 p.m.",
    "Registration for Coldwater Youth Soccer closes August 10. Fees: $85 for U6-U8, $120 for U10-U14, with a $20 sibling discount. Practices are twice weekly; games are Saturdays at Miller Fields.",
    "The Varrow 3-speed blender's jar holds 64 oz and is dishwasher-safe on the top rack. The 1,200-watt motor auto-stops after 60 seconds of continuous run to prevent overheating; wait 2 minutes before restarting.",
    "Tamsin Valley's farmers market runs Saturdays 8 a.m.-noon from the first Saturday in May through October 26. About 35 vendors attend; SNAP is doubled up to $20 per visit at the info tent.",
    "To file a pothole report in Marrowfield, call 311 or use the city app. Include the nearest address and a photo. Crews target repair within 5 business days for holes deeper than 2 inches.",
    "The Corbel Arts Center ceramics course is 8 weeks, Mondays 6-8:30 p.m., for $260 including 25 lb of clay and firing fees. Wheel time outside class is $10 per open-studio session.",
    "Pinehurst Animal Shelter adoption fees: dogs $150, puppies under 6 months $250, cats $75, and pairs of bonded cats $100. All animals are spayed or neutered, microchipped, and vaccinated before release.",
    "Ostrand's leaf collection runs in three sweeps: Oct 20, Nov 10, and Dec 1. Rake leaves to the curb line, not into the street, and keep piles free of branches longer than 4 ft.",
    "The Wexley 40-qt cooler holds ice for up to 4 days at 90°F in testing, weighs 21 lb empty, and has a drain plug with a 3/4-inch hose fitting. The lid gasket is replaceable (part WX-40G, $9).",
    "Mill Pond's free skating rink opens when the ice is 6 inches thick, usually mid-January. Skate rental is $4; the warming hut is open 10 a.m.-8 p.m. on weekends and 3-8 p.m. on weekdays.",
    "Driftwood Inn has 18 rooms. Standard queen rooms are $129 midweek and $169 Friday-Saturday; a two-night minimum applies June-August. Check-in is 3 p.m., checkout 11 a.m., and parking is free.",
    "For the Hartwell Night Run 10K, bib pickup is Friday 4-8 p.m. at Ostrand Station. The course closes 90 minutes after the 8 p.m. start, and there are water stops at miles 2, 4, and 5.5.",
    "Quill & Co. print shop charges $0.12 per black-and-white page and $0.45 per color page for letter size. Binding is $4 per book; orders over 500 pages are ready in 24 hours.",
    "The Briar Lane clinic gives flu shots on a walk-in basis Monday-Friday 9-5. The standard dose is free with most insurance or $32 without; the high-dose vaccine for 65+ is $68 without insurance.",
    "Reed Hollow campground has 54 sites: 30 with electric hookups ($38/night) and 24 tent-only ($22/night). Quiet hours are 10 p.m.-7 a.m., and firewood must be bought on-site to prevent pests.",
    "Ostrand's new water rate is $4.10 per 1,000 gallons for the first 6,000 gallons each month and $5.25 per 1,000 above that. The fixed monthly service charge rose from $11 to $13.50 on July 1.",
    "Calder Bike Co-op's open shop is Tuesday and Thursday 5-9 p.m. Stand time is $5 per hour, and volunteers teach flat repair, brake adjustment, and chain replacement. Used tubes cost $2.",
    "Ferris Hall's 300-seat auditorium rents for $450 for a 4-hour block on weekdays and $700 on weekends. The fee includes the sound system and one technician; projection is $75 extra.",
    "The Harrow 2-person tent packs to 18 x 6 inches and weighs 4 lb 2 oz. It has two doors, two vestibules (9 sq ft each), and a 3,000 mm waterproof rainfly. Pitching takes about 5 minutes.",
]

THIN = [
    "Welcome to our website! We are passionate about delivering the best experience to every customer. Quality and service are at the heart of everything we do. Contact us today to learn more.",
    "Looking for the best pool in town? Look no further. Our pool offers fun for the whole family in a safe and welcoming environment. Come visit us and see why everyone loves it here.",
    "Thermostats are an important part of any home. A good thermostat can help keep you comfortable. If you have questions about your thermostat, it's always a good idea to consult the manual or a professional.",
    "Farm-fresh produce is a wonderful thing. Eating local supports your community and tastes great. Our farm is committed to sustainability and to bringing you the freshest food possible.",
    "Public transit is a convenient way to get around the city. Buses run regularly and serve many destinations. Check the schedule for more details and plan your trip accordingly.",
    "Gardening is a rewarding hobby that brings people together. Community gardens offer a great space to grow your own food and meet your neighbors. Sign up today!",
    "Our dental practice provides comprehensive care for patients of all ages. We use the latest technology and our friendly staff will make you feel right at home. Call to schedule an appointment.",
    "Baking bread at home is a fun and rewarding experience. With a little practice, anyone can make a delicious loaf. Experiment with different flours and techniques to find what you like best.",
    "The library offers many resources for the community. From books to technology, there is something for everyone. Stop by and explore everything the library has to offer.",
    "Hiking is a great way to enjoy nature and stay healthy. There are many beautiful trails in the area suitable for all skill levels. Remember to bring water and enjoy the view!",
    "Prices vary depending on your needs. Contact us for a personalized quote. Our team offers competitive rates and will work with you to find the right solution for your budget.",
    "Hours may vary by season and are subject to change. Please check back later for updated information, or reach out to our office for the latest details on availability.",
    "Recycling is important for our planet. Every little bit helps. Please do your part by recycling whenever you can and encouraging others to do the same.",
    "Our marina is the perfect place to keep your boat. Enjoy beautiful views, friendly staff, and a convenient location. Ask about our slip availability and rates today.",
    "Youth sports teach kids valuable life skills like teamwork and discipline. Our league is a great place for children to have fun and make friends. Registration information coming soon.",
    "This blender is a must-have for any kitchen. It's powerful, easy to use, and looks great on your countertop. Make smoothies, soups, and more in seconds. You'll wonder how you lived without it.",
    "Farmers markets are a vibrant part of community life. Shop local, meet the growers, and discover something new each week. Details about this season's market will be announced.",
    "Road maintenance is a top priority for the city. We are committed to keeping our streets safe and in good condition for all residents and visitors.",
    "Discover your creative side with our art classes. Our experienced instructors make learning fun and accessible. Classes fill up fast, so don't wait — inquire about upcoming sessions now.",
    "Adopting a pet changes lives. Our shelter has many wonderful animals waiting for their forever homes. Visit us to find your new best friend. Fees and requirements may apply.",
    "Fall is a beautiful time of year, but it also means leaves! The city provides seasonal services to help residents. Stay tuned for more information about this year's program.",
    "This cooler is built tough for all your adventures. Keep your food and drinks cold longer. Great for camping, tailgating, fishing, and more. Order yours today while supplies last!",
    "Winter fun awaits! Our rink is a favorite destination for families and friends. Bundle up and come enjoy the season with us. Rental equipment may be available.",
    "Experience comfort and charm at our inn. Whether you're traveling for business or pleasure, our rooms provide everything you need for a relaxing stay. Book now for the best rates.",
    "Join us for an unforgettable race experience! Runners of all levels are welcome. It's a great way to challenge yourself and support a good cause. More details to follow.",
    "Our full-service print shop handles projects of every size with industry-leading turnaround and unbeatable value. Premium quality you can count on, every single time, guaranteed.",
    "Staying healthy is important, and vaccines are one of the best ways to protect yourself and your loved ones. Talk to your healthcare provider about which vaccines are right for you.",
    "Camping is a classic way to reconnect with nature. Our campground offers a peaceful setting and a range of options for every type of camper. Reserve your spot for the season.",
    "Water is a precious resource. The city is working hard to provide reliable, high-quality water service at a fair price. Rate information is available upon request.",
    "Whether you're a beginner or an expert, our community bike shop welcomes you. Learn new skills, meet fellow cyclists, and keep your ride in top shape. Everyone is welcome!",
]


def build_substance() -> None:
    items = [("yes", t) for t in SUBSTANTIVE] + [("no", t) for t in THIN]
    rng.shuffle(items)
    rows = []
    yes_seen = no_seen = 0
    for i, (label, text) in enumerate(items):
        # first 10 of each class -> calibrate split, rest -> score split (20 + 20 scored)
        if label == "yes":
            split = "calibrate" if yes_seen < 10 else "score"
            yes_seen += 1
        else:
            split = "calibrate" if no_seen < 10 else "score"
            no_seen += 1
        rows.append({"id": f"sub-{i:03d}", "input": {"text": text}, "expected": label, "meta": {"split": split}})
    write(ROOT / "substance" / "substance.jsonl", rows)


# ---------------------------------------------------------------------------------------
# 2. qa_judge: judge calibration. Each row = question + reference + a candidate answer whose
#    pass/fail label is fixed by how the candidate was constructed:
#      paraphrase           -> pass   (same facts, different words)
#      correct+extra        -> pass   (correct, adds harmless true context)
#      wrong_number         -> fail   (one quantity changed)
#      wrong_entity         -> fail   (fluent, but names the wrong thing)
#      evasive              -> fail   (does not answer)
#      terse_correct        -> pass   (bare correct answer, no explanation)
#      correct_plus_false_claim -> fail (right answer + one false side claim)
#    Questions are stable, widely documented general-knowledge facts.
# ---------------------------------------------------------------------------------------

QA = [
    ("At sea level, what is the boiling point of pure water in degrees Celsius?", "100 °C",
     "Pure water boils at 100 degrees Celsius at sea level.",
     "It boils at 100 °C at sea level; at higher altitudes it boils at a lower temperature.",
     "Pure water boils at 90 degrees Celsius at sea level.",
     "At sea level, water reaches its freezing point at 0 °C, which is when it changes state.",
     "It depends on many factors, so there isn't a single answer.",
     '100.', 'Water boils at 100 °C at sea level, which is the same as 100 °F.'),
    ("How many sides does a hexagon have?", "6",
     "A hexagon has six sides.",
     "Six. A regular hexagon also has six equal interior angles of 120 degrees.",
     "A hexagon has eight sides.",
     "A pentagon is the polygon with sides like this; it's a common shape in architecture.",
     "Polygons come in many forms, and the answer varies by definition.",
     '6', 'A hexagon has six sides, the same number of sides as an octagon.'),
    ("What planet is closest to the Sun?", "Mercury",
     "Mercury is the planet nearest the Sun.",
     "Mercury; it completes an orbit in about 88 Earth days.",
     "Mercury is the third planet from the Sun.",
     "Venus is the closest planet to the Sun.",
     "The solar system is vast and planets move, so it's hard to say.",
     'Mercury.', 'Mercury is closest to the Sun, and it is also the largest planet in the solar system.'),
    ("How many minutes are in 2.5 hours?", "150 minutes",
     "2.5 hours equals 150 minutes.",
     "150 minutes, since each hour has 60 minutes.",
     "2.5 hours equals 125 minutes.",
     "2.5 hours is 9,000 seconds, which is a common unit of time.",
     "Time conversion can be tricky; consider using a calculator.",
     '150.', '2.5 hours is 150 minutes, or 15,000 seconds.'),
    ("What gas do plants primarily absorb from the air for photosynthesis?", "Carbon dioxide (CO2)",
     "Plants take in carbon dioxide for photosynthesis.",
     "Carbon dioxide; they use it with water and light to make sugars, releasing oxygen.",
     "Plants primarily absorb carbon dioxide at night only, and none during the day.",
     "Plants primarily absorb nitrogen from the air for photosynthesis.",
     "Plants interact with many gases in complex ways.",
     'CO2.', 'Carbon dioxide, which makes up about 21% of the atmosphere.'),
    ("How many continents are there in the most commonly taught model in English-speaking countries?", "7",
     "That model has seven continents.",
     "Seven: Africa, Antarctica, Asia, Australia/Oceania, Europe, North America, South America.",
     "That model has five continents.",
     "There are seven oceans in that model.",
     "Different people count continents differently, so I can't say.",
     'Seven.', 'Seven continents, of which Australia is the largest.'),
    ("What is the chemical symbol for sodium?", "Na",
     "Sodium's chemical symbol is Na.",
     "Na, from its Latin name natrium.",
     "Sodium's chemical symbol is Na2.",
     "Sodium's chemical symbol is S.",
     "Chemical symbols are listed on the periodic table.",
     'Na', 'Na; sodium is a noble gas.'),
    ("How many bits are in one byte?", "8",
     "One byte is eight bits.",
     "Eight bits; a byte can therefore represent 256 distinct values.",
     "One byte is sixteen bits.",
     "One byte is 1,024 kilobytes.",
     "It depends on the computer architecture you are asking about.",
     '8', 'Eight bits, so one byte can represent 1,024 distinct values.'),
    ("What is the largest ocean on Earth?", "The Pacific Ocean",
     "The Pacific is Earth's largest ocean.",
     "The Pacific Ocean, which covers more area than all land combined.",
     "The Pacific is the largest ocean, at about half the size of the Atlantic.",
     "The Atlantic Ocean is the largest ocean on Earth.",
     "Oceans are all connected, so they are really one big ocean.",
     'The Pacific.', 'The Pacific Ocean, which lies between Europe and North America.'),
    ("How many degrees are in a right angle?", "90",
     "A right angle measures 90 degrees.",
     "Ninety degrees, a quarter of a full turn.",
     "A right angle measures 180 degrees.",
     "A right angle is the angle in an equilateral triangle.",
     "Angles can be measured in degrees or radians.",
     '90°', '90 degrees, which is half of a full turn.'),
    ("What is the freezing point of water in degrees Fahrenheit at standard pressure?", "32 °F",
     "Water freezes at 32 degrees Fahrenheit.",
     "32 °F, which is 0 °C.",
     "Water freezes at 0 degrees Fahrenheit.",
     "Water boils at 212 °F, which is the key reference temperature.",
     "That depends on the thermometer you use.",
     '32', '32 °F, which equals 10 °C.'),
    ("How many legs does an insect have?", "6",
     "Insects have six legs.",
     "Six, attached to the thorax.",
     "Insects have eight legs.",
     "Spiders are insects with legs attached to the abdomen.",
     "Bugs come in so many varieties that there's no single number.",
     '6', 'Six legs, attached to the abdomen.'),
    ("What is the square root of 81?", "9",
     "The square root of 81 is 9.",
     "9, because 9 × 9 = 81 (−9 is also a square root).",
     "The square root of 81 is 8.",
     "81 squared is 6,561.",
     "Square roots can be computed with a calculator.",
     '9', '9, because 9 × 9 = 81, and 9 is also a prime number.'),
    ("In which organ of the human body is insulin produced?", "The pancreas",
     "Insulin is made in the pancreas.",
     "The pancreas, specifically by beta cells in the islets of Langerhans.",
     "Insulin is made in the pancreas, which is located in the skull.",
     "Insulin is produced in the liver.",
     "Hormones are produced throughout the body.",
     'Pancreas.', 'The pancreas, which is part of the respiratory system.'),
    ("How many sides does a triangle have?", "3",
     "A triangle has three sides.",
     "Three sides, and its interior angles sum to 180 degrees.",
     "A triangle has four sides.",
     "A square has sides of equal length.",
     "Shapes are a fascinating topic in geometry.",
     '3', 'Three sides, and its interior angles add up to 360 degrees.'),
    ("What is 15% of 200?", "30",
     "15 percent of 200 is 30.",
     "30, since 10% is 20 and 5% is 10.",
     "15 percent of 200 is 35.",
     "200 is 15% more than 174.",
     "Percentages are easiest with a calculator.",
     '30', '30, which is the same as 10% of 200.'),
]

KINDS = [("paraphrase", "pass"), ("correct_extra", "pass"), ("wrong_number", "fail"),
         ("wrong_entity", "fail"), ("evasive", "fail"),
         # the two classic judge traps: too strict on terse-but-right, too lenient on right-plus-one-false-claim
         ("terse_correct", "pass"), ("correct_plus_false_claim", "fail")]


def build_qa_judge() -> None:
    rows = []
    for qi, (q, ref, *cands) in enumerate(QA):
        for (kind, label), cand in zip(KINDS, cands):
            rows.append({
                "id": f"qa-{qi:02d}-{kind}",
                "input": {"question": q, "reference": ref, "candidate": cand},
                "expected": label,
                "meta": {"kind": kind},
            })
    rng.shuffle(rows)
    write(ROOT / "qa_judge" / "qa_judge.jsonl", rows)

    # pairwise: three "systems" per question; by construction correct ones should beat wrong ones.
    pair_rows = []
    for qi, (q, _ref, para, extra, wrong_num, *_rest) in enumerate(QA):
        pair_rows.append({
            "id": f"pw-{qi:02d}",
            "input": {"question": q, "answers": {"concise": para, "detailed": extra, "confident_wrong": wrong_num}},
            "expected": {"loser": "confident_wrong"},
        })
    write(ROOT / "qa_judge" / "pairwise.jsonl", pair_rows)


# ---------------------------------------------------------------------------------------
# 3. retrieval: a tiny fictional ops-runbook corpus + queries with gold doc ids.
# ---------------------------------------------------------------------------------------

CORPUS = {
    "rb-backup": "Nightly database backup runs at 02:00 and is kept for 14 days. Restore with the restore-snapshot procedure.",
    "rb-restore": "Restore-snapshot procedure: stop writers, pick a snapshot by timestamp, restore to a scratch database, verify row counts, then swap.",
    "rb-cert": "TLS certificates renew automatically 30 days before expiry. If renewal fails, the expiry alarm fires 7 days out.",
    "rb-disk": "Disk usage alarm fires at 85 percent. First check log rotation, then old build artifacts, then temp upload folders.",
    "rb-logs": "Log rotation keeps 7 compressed daily files per service. Rotation failures usually mean a full disk or a permissions change.",
    "rb-deploy": "Deploys run from the main branch after tests pass. A deploy can be rolled back by redeploying the previous build.",
    "rb-rollback": "Rollback: find the last green build, redeploy it, then confirm the version endpoint reports the old build id.",
    "rb-queue": "If the job queue backs up, check for one long-running job starving the workers before adding more workers.",
    "rb-cache": "Cache keys expire after one hour. Purging the whole cache during peak traffic causes a load spike on the database.",
    "rb-dns": "DNS changes can take up to 48 hours to propagate. Lower the record TTL a day before a planned migration.",
    "rb-rate": "API rate limit is 100 requests per minute per key. Bursts above the limit receive HTTP 429 with a retry-after header.",
    "rb-oncall": "On-call rotation changes every Monday at 09:00. Escalate to the secondary after 15 minutes without acknowledgement.",
    "rb-secrets": "Secrets live in the secrets manager, never in the repository. Rotate a leaked key first, then purge it from history.",
    "rb-migrate": "Schema migrations run before the new code starts. Destructive migrations require a two-step expand then contract rollout.",
    "rb-memory": "Memory alarm at 90 percent. Look for a worker that never releases memory; restarting the worker is the short-term fix.",
    "rb-sleep": "Build hosts must not sleep. A sleeping host silently stops running scheduled jobs until someone wakes it.",
    "rb-heartbeat": "Every scheduled job writes a heartbeat. A separate monitor alarms when a heartbeat is older than twice the schedule interval.",
    "rb-flaky": "A flaky test is quarantined with an owner and a ticket, never deleted. Quarantine expires after 14 days.",
    "rb-cost": "Monthly cloud cost review compares spend to the budget; any service growing more than 20 percent month over month gets a ticket.",
    "rb-index": "Search index rebuild takes about 40 minutes. Queries during a rebuild read the previous index until the swap.",
}

QUERIES = [
    ("How long are database backups kept?", ["rb-backup"], True),
    ("Steps to restore the database from a snapshot", ["rb-restore", "rb-backup"], True),
    ("What happens if certificate renewal fails?", ["rb-cert"], True),
    ("Disk is almost full, what should I check first?", ["rb-disk", "rb-logs"], True),
    ("How do I roll back a bad deploy?", ["rb-rollback", "rb-deploy"], True),
    ("The job queue is backed up", ["rb-queue"], False),
    ("Is it safe to purge the cache at peak?", ["rb-cache"], False),
    ("When should I lower the DNS TTL?", ["rb-dns"], False),
    ("What does HTTP 429 mean for our API?", ["rb-rate"], False),
    ("Who do I escalate to if on-call doesn't respond?", ["rb-oncall"], True),
    ("A key leaked into git, what do I do?", ["rb-secrets"], True),
    ("How should I run a destructive schema change?", ["rb-migrate"], True),
    ("Worker memory keeps climbing", ["rb-memory"], False),
    ("Scheduled jobs stopped running overnight on the build machine", ["rb-sleep", "rb-heartbeat"], True),
    ("What do we do with a flaky test?", ["rb-flaky"], False),
    ("Spend jumped on one service this month", ["rb-cost"], False),
    ("Can people search while the index rebuilds?", ["rb-index"], False),
]


def build_retrieval() -> None:
    write(ROOT / "retrieval" / "corpus.jsonl", [{"id": k, "text": v} for k, v in CORPUS.items()])
    write(ROOT / "retrieval" / "queries.jsonl", [
        {"id": f"q-{i:02d}", "input": q, "expected": gold, "meta": {"critical": crit}}
        for i, (q, gold, crit) in enumerate(QUERIES)
    ])


if __name__ == "__main__":
    build_substance()
    build_qa_judge()
    build_retrieval()
    print("fixtures rebuilt under", ROOT)
