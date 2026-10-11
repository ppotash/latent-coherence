"""Hand-written gold outputs for the decomposition stage.

Used (a) as few-shot examples in the Gemini prompt and (b) as a fixture backend so the
pipeline can be tested end-to-end without API access. uids are real HoVer train uids.
"""

GOLD = {
    "52e11a80": {
        "claim": "The American publishing company headquartered in Scribner Building publishes the books of the author Donald Richard \"Don\" DeLillo whose books cover subjects such as the Cold War, mathematics, politics, and economics, and global terrorism.",
        "output": {
            "usable": True,
            "reason": "Conjunction of facts about a publisher (unnamed in the claim, so it is given an invented name), its headquarters, its author, and the author's subjects.",
            "entity_map": [
                {"original": "(unnamed publisher headquartered in the Scribner Building)", "fictional": "Ostrander & Pell", "kind": "organization"},
                {"original": "Scribner Building", "fictional": "the Halvard Building", "kind": "place"},
                {"original": "Donald Richard \"Don\" DeLillo", "fictional": "Corwin Ashby", "kind": "person"},
            ],
            "subclaims": [
                {"original": "The publisher is an American publishing company headquartered in the Scribner Building.",
                 "fictional": "Ostrander & Pell is an American publishing company headquartered in the Halvard Building.",
                 "fictional_negation": "It is not the case that Ostrander & Pell is an American publishing company headquartered in the Halvard Building."},
                {"original": "The publisher publishes the books of Don DeLillo.",
                 "fictional": "Ostrander & Pell publishes the books of the author Corwin Ashby.",
                 "fictional_negation": "Ostrander & Pell does not publish the books of the author Corwin Ashby."},
                {"original": "Don DeLillo's books cover subjects such as the Cold War, mathematics, politics, economics, and global terrorism.",
                 "fictional": "Corwin Ashby's books cover subjects such as the Cold War, mathematics, politics, economics, and global terrorism.",
                 "fictional_negation": "It is not the case that Corwin Ashby's books cover subjects such as the Cold War, mathematics, politics, economics, and global terrorism."},
            ],
            "fictional_claim_explicit": "Ostrander & Pell, an American publishing company headquartered in the Halvard Building, publishes the books of the author Corwin Ashby, whose books cover subjects such as the Cold War, mathematics, politics, economics, and global terrorism.",
            "fictional_claim_descriptive": "The American publishing company headquartered in the Halvard Building publishes the books of the author Corwin Ashby, whose books cover subjects such as the Cold War, mathematics, politics, economics, and global terrorism.",
            "bridge_index": 1,
        },
    },
    "cac2e9c0": {
        "claim": "The star of 7 Lives also plays the role of Liz in the 2004 British horror comedy film about a zombie uprising.",
        "output": {
            "usable": True,
            "reason": "Conjunction linking an unnamed actor (the bridge entity, given an invented name) to two films.",
            "entity_map": [
                {"original": "7 Lives", "fictional": "Nine Doors", "kind": "work"},
                {"original": "(unnamed star of 7 Lives)", "fictional": "Dara Fenwick", "kind": "person"},
                {"original": "(unnamed 2004 British horror comedy about a zombie uprising)", "fictional": "Grave Laughter", "kind": "work"},
            ],
            "subclaims": [
                {"original": "The actor is the star of 7 Lives.",
                 "fictional": "Dara Fenwick is the star of the film Nine Doors.",
                 "fictional_negation": "Dara Fenwick is not the star of the film Nine Doors."},
                {"original": "The actor plays the role of Liz in the zombie film.",
                 "fictional": "Dara Fenwick plays the role of Liz in the film Grave Laughter.",
                 "fictional_negation": "Dara Fenwick does not play the role of Liz in the film Grave Laughter."},
                {"original": "The zombie film is a 2004 British horror comedy about a zombie uprising.",
                 "fictional": "Grave Laughter is a 2004 British horror comedy film about a zombie uprising.",
                 "fictional_negation": "It is not the case that Grave Laughter is a 2004 British horror comedy film about a zombie uprising."},
            ],
            "fictional_claim_explicit": "Dara Fenwick, the star of the film Nine Doors, plays the role of Liz in Grave Laughter, which is a 2004 British horror comedy film about a zombie uprising.",
            "fictional_claim_descriptive": "The star of Nine Doors also plays the role of Liz in the 2004 British horror comedy film about a zombie uprising.",
            "bridge_index": 1,
        },
    },
    "nba_barnes": {
        "claim": "The NBA Draft in which Harrison Bryce Jordan Barnes was selected by the Golden State Warriors in the first round of the draft with the seventh overall pick was held in the Prudential Center in Newark.",
        "output": {
            "usable": True,
            "reason": "Two facts joined through a bridge entity (the draft), which is given an explicit invented name.",
            "entity_map": [
                {"original": "NBA", "fictional": "Continental Basketball Association (CBA)", "kind": "organization"},
                {"original": "(the draft in which Barnes was picked)", "fictional": "the 2012 CBA Draft", "kind": "event"},
                {"original": "Harrison Bryce Jordan Barnes", "fictional": "Tavin Rourke", "kind": "person"},
                {"original": "Golden State Warriors", "fictional": "the Bayport Gulls", "kind": "organization"},
                {"original": "Prudential Center", "fictional": "the Calloway Arena", "kind": "place"},
                {"original": "Newark", "fictional": "Westbrook", "kind": "place"},
            ],
            "subclaims": [
                {"original": "Barnes was selected by the Golden State Warriors with the seventh overall pick in the first round of the draft.",
                 "fictional": "Tavin Rourke was selected by the Bayport Gulls with the seventh overall pick in the first round of the 2012 CBA Draft.",
                 "fictional_negation": "Tavin Rourke was not selected by the Bayport Gulls with the seventh overall pick in the first round of the 2012 CBA Draft."},
                {"original": "That draft was held in the Prudential Center in Newark.",
                 "fictional": "The 2012 CBA Draft was held in the Calloway Arena in Westbrook.",
                 "fictional_negation": "The 2012 CBA Draft was not held in the Calloway Arena in Westbrook."},
            ],
            "fictional_claim_explicit": "The 2012 CBA Draft, in which Tavin Rourke was selected by the Bayport Gulls with the seventh overall pick in the first round, was held in the Calloway Arena in Westbrook.",
            "fictional_claim_descriptive": "The CBA Draft in which Tavin Rourke was selected by the Bayport Gulls in the first round with the seventh overall pick was held in the Calloway Arena in Westbrook.",
            "bridge_index": 0,
        },
    },
    "neg_notboth": {
        "claim": "Jeff Ragsdale and Millard Webb are not both authors.",
        "output": {"usable": False,
                   "reason": "Negated conjunction ('not both'): the claim is true when either fact fails, so it is not a conjunction of atomic facts.",
                   "entity_map": [], "subclaims": [], "fictional_claim_explicit": "",
                   "fictional_claim_descriptive": "", "bridge_index": -1},
    },
    "neg_compare": {
        "claim": "Louis XIV formed earlier than the band whose album was Up from Below.",
        "output": {"usable": False,
                   "reason": "Comparison of two dates: truth depends on a relation between two quantities, not on a conjunction of independent facts.",
                   "entity_map": [], "subclaims": [], "fictional_claim_explicit": "",
                   "fictional_claim_descriptive": "", "bridge_index": -1},
    },
}

# real HoVer uid prefixes -> full uids are resolved at load time; the two synthetic keys are
# few-shot-only examples written in the style of HoVer claims.
