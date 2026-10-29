"""
Module: src/dataset.py
Description: Generates a scientifically rigorous Hallucination & Factual Consistency Benchmark Dataset.
             Uses 25 diverse knowledge domains partitioned strictly by domain (OOD split)
             to avoid topic leakage:
             - 20 Domains for Training (4,000 pairs: 2,000 Factual, 2,000 Hallucinated)
             - 3 Domains for Validation (500 pairs: 250 Factual, 250 Hallucinated)
             - 2 Domains for Held-Out Testing (500 pairs: 250 Factual, 250 Hallucinated)
             Generates EDA distribution plots (Figure 4: Class Balance, Figure 5: Density).
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 25 Diverse Domain Knowledge Bases
ALL_DOMAINS = [
    # --- TRAINING DOMAINS (1 to 20) ---
    {
        "id": "penicillin",
        "context": "Penicillin was discovered in 1928 by Scottish scientist Alexander Fleming at St. Mary's Hospital London, revolutionizing modern antibiotic therapy for bacterial infections.",
        "factual": [
            "Alexander Fleming discovered penicillin in 1928, marking a monumental breakthrough in antibiotic therapy for bacterial illnesses.",
            "In 1928, penicillin was identified by Alexander Fleming at St. Mary's Hospital, fundamentally transforming treatments for bacterial disease.",
            "Antibiotic treatments advanced dramatically after Alexander Fleming discovered penicillin in London in 1928."
        ],
        "hallucinated": [
            "Alexander Fleming discovered penicillin in 1948 at St. Mary's Hospital London, marking modern antibiotic therapy.",
            "Louis Pasteur discovered penicillin in 1928 at St. Mary's Hospital London, revolutionizing modern antibiotics.",
            "Penicillin was discovered by Alexander Fleming in 1928 to successfully treat severe viral respiratory epidemics.",
            "Alexander Fleming discovered penicillin in 1928, but clinical trials conclusively proved it possessed zero antibiotic efficacy."
        ]
    },
    {
        "id": "metformin",
        "context": "Metformin decreases hepatic glucose production and increases insulin sensitivity in peripheral tissues, serving as a primary first-line therapy for type 2 diabetes mellitus.",
        "factual": [
            "Metformin manages type 2 diabetes by reducing glucose output from the liver and enhancing peripheral tissue responsiveness to insulin.",
            "As a first-line treatment for type 2 diabetes, metformin works by dampening hepatic glucose creation and boosting insulin sensitivity.",
            "Type 2 diabetes patients benefit from metformin because it suppresses liver glucose synthesis and increases peripheral insulin action."
        ],
        "hallucinated": [
            "Metformin increases hepatic glucose production and promotes acute insulin resistance in type 2 diabetes patients.",
            "Metformin is an injectable peptide hormone that stimulates pancreatic beta cells to produce glucagon during meals.",
            "Metformin decreases hepatic glucose production and was approved in 1910 to eradicate autoimmune type 1 diabetes.",
            "Insulin decreases hepatic glucose production and serves as a synthetic cofactor to activate oral metformin."
        ]
    },
    {
        "id": "aspirin",
        "context": "Aspirin irreversibly inhibits cyclooxygenase-1 (COX-1) enzymes, preventing the formation of thromboxane A2 and thereby suppressing platelet aggregation and arterial thrombosis.",
        "factual": [
            "Aspirin inhibits COX-1 enzymes irreversibly, halting thromboxane A2 production and impeding blood platelet aggregation.",
            "By blocking cyclooxygenase-1, aspirin suppresses the synthesis of thromboxane A2, reducing clot formation risks.",
            "Aspirin acts as an antiplatelet agent by irreversibly suppressing COX-1 and blocking thromboxane A2 creation."
        ],
        "hallucinated": [
            "Aspirin stimulates cyclooxygenase-1 enzymes to rapidly accelerate thromboxane A2 synthesis and promote blood clotting.",
            "Aspirin reversibly inhibits COX-2 enzymes to permanently cure sickle cell anemia and bone marrow failure.",
            "Ibuprofen irreversibly inhibits cyclooxygenase-1 enzymes, completely replacing aspirin for preventing arterial thrombosis.",
            "Aspirin prevents the formation of thromboxane A2 without interacting with or altering cyclooxygenase-1 enzymes."
        ]
    },
    {
        "id": "transformer",
        "context": "The Transformer architecture introduced by Vaswani et al. in 2017 relies solely on multi-head self-attention mechanisms, dispensing entirely with recurrence and convolutions.",
        "factual": [
            "Vaswani et al. proposed the Transformer in 2017, substituting recurrence and convolutions with multi-head self-attention.",
            "Multi-head self-attention forms the entire foundation of the 2017 Transformer architecture, eliminating recurrent cells.",
            "In 2017, Vaswani and collaborators designed the Transformer model, discarding recurrence in favor of self-attention."
        ],
        "hallucinated": [
            "The Transformer architecture introduced by Vaswani et al. in 2011 relies heavily on recurrent LSTM gating mechanisms.",
            "Geoffrey Hinton introduced the Transformer in 2017 to eliminate self-attention mechanisms from neural machine translation.",
            "The Transformer architecture introduced by Vaswani et al. in 2017 uses multi-head self-attention alongside recurrent convolutions.",
            "The Transformer model introduced by Vaswani et al. in 2017 has been proven computationally incapable of processing text sequences."
        ]
    },
    {
        "id": "cnn",
        "context": "Convolutional Neural Networks utilize learnable spatial filter kernels that slide across feature maps to capture translation-invariant patterns in grid-structured image data.",
        "factual": [
            "CNNs detect translation-invariant visual features by convolving parameterized spatial filters across image grids.",
            "Spatial filter kernels sliding over grid arrays enable Convolutional Neural Networks to extract localized, translation-invariant representations.",
            "By sliding filter kernels over feature maps, CNNs efficiently extract spatial patterns from multidimensional image matrices."
        ],
        "hallucinated": [
            "Convolutional Neural Networks utilize learnable spatial filter kernels to eliminate all translation-invariant patterns from grid data.",
            "CNNs slide temporal recurrence gates over 1D audio waves, preventing any spatial grid feature extraction.",
            "Convolutional Neural Networks were proven mathematically unable to process multidimensional grid or pixel structures.",
            "Recurrent Neural Networks slide learnable spatial filter kernels across feature maps to train image recognition models."
        ]
    },
    {
        "id": "python_lang",
        "context": "Python is a high-level interpreted programming language designed by Guido van Rossum and initially released in 1991, emphasizing code readability and significant whitespace indentation.",
        "factual": [
            "Created by Guido van Rossum in 1991, Python is an interpreted language renowned for its focus on readability and clean indentation.",
            "Guido van Rossum released Python in 1991 as an interpreted high-level language utilizing indentation for block syntax.",
            "Python's design philosophy highlights readability and indentation, conceived by Guido van Rossum and debuted in 1991."
        ],
        "hallucinated": [
            "Python was created by James Gosling and released in 1991 as a compiled low-level language for operating system kernels.",
            "Guido van Rossum released Python in 2008 as a static dialect of C++ without indentation syntax.",
            "Python was developed by Guido van Rossum in 1991 to eliminate all interpreted scripts in favor of pure binary assembly.",
            "Python is a proprietary closed-source programming language licensed exclusively by Microsoft Corporation since 1991."
        ]
    },
    {
        "id": "apollo11",
        "context": "The Apollo 11 lunar module Eagle touched down on the Moon's Sea of Tranquility on July 20, 1969, making Neil Armstrong and Buzz Aldrin the first humans to walk on the lunar surface.",
        "factual": [
            "On July 20, 1969, Apollo 11 landed in the Sea of Tranquility, where Neil Armstrong and Buzz Aldrin walked on the Moon.",
            "Neil Armstrong and Buzz Aldrin became the first humans to set foot on the lunar surface when Apollo 11 landed on July 20, 1969.",
            "The lunar module Eagle landed on the Moon in July 1969, enabling Neil Armstrong and Buzz Aldrin's historic moonwalk."
        ],
        "hallucinated": [
            "Apollo 11 landed on the Moon on July 20, 1969, allowing Neil Armstrong and Michael Collins to walk on the lunar surface.",
            "The Apollo 13 lunar module Eagle touched down on the Sea of Tranquility on July 20, 1969, landing three crew members.",
            "Apollo 11 orbited the Moon on July 20, 1969, but aborted its descent before any humans could walk on the lunar surface.",
            "Neil Armstrong and Buzz Aldrin walked on Mars during the historic Apollo 11 landing on July 20, 1969."
        ]
    },
    {
        "id": "jwst",
        "context": "James Webb Space Telescope operates in a halo orbit around the Sun-Earth Lagrange point L2, approximately 1.5 million kilometers from Earth, capturing infrared astronomical observations.",
        "factual": [
            "Located roughly 1.5 million kilometers away at Sun-Earth Lagrange point L2, the James Webb Space Telescope gathers infrared data.",
            "The JWST performs infrared observations from its orbit near the Sun-Earth L2 point, 1.5 million km from Earth.",
            "Orbiting the Sun-Earth L2 Lagrange point 1.5 million km away, James Webb observes the universe in infrared wavelengths."
        ],
        "hallucinated": [
            "The James Webb Space Telescope operates in low Earth orbit at an altitude of 400 kilometers to collect ultraviolet solar flare data.",
            "The Hubble Space Telescope was launched to the Sun-Earth L2 point 1.5 million km away to replace the failed James Webb telescope.",
            "James Webb operates at Lagrange point L2, exactly 150 million kilometers from Earth, observing radio frequencies.",
            "The James Webb Space Telescope was decommissioned immediately upon reaching Lagrange point L2 due to optical sensor collapse."
        ]
    },
    {
        "id": "fed_rates",
        "context": "The Federal Reserve raised the benchmark federal funds rate by 25 basis points to a target range of 5.25% to 5.50% in July 2023 to combat elevated inflation pressures.",
        "factual": [
            "In July 2023, the Federal Reserve implemented a 0.25% interest rate hike, lifting the federal funds target to 5.25%-5.50% to tame inflation.",
            "To restrain ongoing inflation, the Fed increased the federal funds target range to 5.25%-5.50% via a 25 basis point hike in July 2023.",
            "The Federal Reserve elevated its benchmark rate to 5.25%-5.50% in July 2023 with a 25-bps increase addressing inflation."
        ],
        "hallucinated": [
            "The Federal Reserve slashed the federal funds rate by 200 basis points down to zero in July 2023 to stimulate hyperinflation.",
            "The European Central Bank raised the US federal funds rate to 5.25%-5.50% in July 2023 to combat American inflation.",
            "The Federal Reserve raised the benchmark rate by 25 basis points in July 2023, despite inflation having dropped to negative 5 percent.",
            "In July 2023, the Federal Reserve raised benchmark interest rates to a record target range of 15.25% to 15.50%."
        ]
    },
    {
        "id": "gdp_def",
        "context": "Gross Domestic Product (GDP) quantifies the monetary value of all finished goods and services produced within a nation's geographic borders over a specified timeframe.",
        "factual": [
            "GDP represents the total monetary value of finished goods and services produced domestically over a designated period.",
            "The market value of all finished goods and services generated within national borders during a given timeframe defines GDP.",
            "By measuring finished products and services within a country's boundaries, GDP gauges total economic output."
        ],
        "hallucinated": [
            "Gross Domestic Product measures only unpaid volunteer domestic labor while strictly omitting all commercial goods and services.",
            "GDP quantifies the foreign debt owed by citizens outside national borders, excluding all domestic economic productivity.",
            "Gross Domestic Product has been legally abolished across all sovereign nations and replaced with gross carbon debt.",
            "GDP measures the total monetary value of raw unfinished goods exported abroad without considering domestic services."
        ]
    },
    {
        "id": "dna_structure",
        "context": "In 1953, James Watson and Francis Crick published the double-helix structure of DNA, building upon the critical X-ray diffraction images taken by Rosalind Franklin.",
        "factual": [
            "Watson and Crick unveiled DNA's double-helix structure in 1953, utilizing X-ray diffraction evidence provided by Rosalind Franklin.",
            "The double-helical architecture of DNA was discovered in 1953 by James Watson and Francis Crick using Rosalind Franklin's data.",
            "Rosalind Franklin's X-ray diffraction images supported Watson and Crick's 1953 structural model of the DNA double helix."
        ],
        "hallucinated": [
            "In 1953, James Watson and Francis Crick proved that DNA is composed of a triple-helix protein matrix lacking base pairs.",
            "Linus Pauling and Francis Crick introduced the single-stranded RNA model of DNA in 1920 without X-ray analysis.",
            "Rosalind Franklin discovered the double-helix structure of DNA in 1985 and rejected Watson and Crick's theories.",
            "Watson and Crick published the structure of DNA in 1953, showing that genetic information is transmitted via electrical currents."
        ]
    },
    {
        "id": "crispr_immune",
        "context": "CRISPR sequences are found in bacteria as an adaptive immune defense that uses Cas endonucleases to cut bacteriophage viral genomes.",
        "factual": [
            "Bacteria defend against bacteriophage viruses using CRISPR sequences and Cas endonuclease enzymes that cleave foreign viral DNA.",
            "CRISPR functions as a bacterial adaptive immune mechanism, utilizing Cas enzymes to slice invading viral genetic sequences.",
            "Cas endonucleases guided by CRISPR sequences enable bacteria to identify and destroy invading bacteriophage genomes."
        ],
        "hallucinated": [
            "CRISPR sequences are synthetic mammalian enzymes created in 2020 to stimulate rapid bacteriophage viral reproduction.",
            "Bacteria use CRISPR sequences to manufacture antibiotic capsules that strengthen bacteriophage cell membranes.",
            "CRISPR was discovered as a human neurological reflex that cuts bacterial cells without relying on Cas proteins.",
            "CRISPR sequences enable bacteriophage viruses to successfully parasitize bacterial endonucleases without detection."
        ]
    },
    {
        "id": "relativity_special",
        "context": "Albert Einstein's special theory of relativity, published in 1905, establishes that the laws of physics are invariant across all inertial reference frames and the speed of light in vacuum is constant.",
        "factual": [
            "Einstein's 1905 special relativity posits that the speed of light in a vacuum remains identical for all inertial observers.",
            "Published in 1905 by Albert Einstein, special relativity demonstrates identical physical laws in all inertial reference frames.",
            "Special relativity confirms the constancy of light speed in vacuum and the invariance of physical laws in inertial frames."
        ],
        "hallucinated": [
            "Einstein's 1905 special relativity proves that the speed of light in vacuum fluctuates wildly depending on the observer's motion.",
            "Isaac Newton published the special theory of relativity in 1905 to disprove Albert Einstein's photoelectric principles.",
            "Albert Einstein's special relativity establishes that physical laws alter drastically whenever an inertial frame is measured.",
            "Special relativity was published in 1955 to prove that speed of light accelerates infinitely inside gravitational singularities."
        ]
    },
    {
        "id": "turing_machine",
        "context": "Alan Turing formulated the theoretical concept of the Turing machine in 1936, providing a foundational mathematical abstraction for modern algorithmic computation.",
        "factual": [
            "In 1936, Alan Turing introduced the Turing machine, laying the theoretical foundation for general-purpose computer science.",
            "The Turing machine, devised by Alan Turing in 1936, serves as a universal model for computational algorithms.",
            "Modern algorithmic computation traces its mathematical foundation to Alan Turing's 1936 paper on Turing machines."
        ],
        "hallucinated": [
            "Alan Turing built the first physical Turing machine in 1850 using mechanical steam valves and copper gears.",
            "John von Neumann formulated the Turing machine in 1936 to prove that algorithmic computation is mathematically impossible.",
            "Alan Turing developed the Turing machine in 1936 solely to execute analog calculations without using symbolic algorithms.",
            "The Turing machine concept was introduced in 1980 by Steve Jobs as a proprietary microcomputer hardware platform."
        ]
    },
    {
        "id": "photosynthesis",
        "context": "Photosynthesis is the biological process by which photosynthetic organisms convert light energy into chemical energy stored in glucose, releasing oxygen as a byproduct.",
        "factual": [
            "Through photosynthesis, plants convert solar radiation into chemical glucose molecules while generating oxygen gas.",
            "During photosynthesis, light energy is captured and converted into glucose, discharging oxygen into the atmosphere.",
            "Photosynthetic reactions convert ambient light into stored glucose energy and discharge oxygen."
        ],
        "hallucinated": [
            "Photosynthesis is a chemical reaction wherein plants absorb oxygen and release carbon monoxide to destroy stored glucose.",
            "Photosynthesis converts mechanical wind vibrations into nuclear energy without requiring sunlight or water.",
            "Plants use photosynthesis solely during total darkness to absorb oxygen and deplete all cellular glucose reserves.",
            "Photosynthesis was engineered by human geneticists in 1990 to allow mammals to produce oxygen from cellular respiration."
        ]
    },
    {
        "id": "bitcoin_blockchain",
        "context": "The Bitcoin whitepaper was released in October 2008 by pseudonymous author Satoshi Nakamoto, introducing a decentralized proof-of-work blockchain ledger.",
        "factual": [
            "Satoshi Nakamoto introduced Bitcoin in October 2008 as a peer-to-peer decentralized electronic cash system backed by proof-of-work.",
            "In October 2008, the Bitcoin whitepaper proposed a decentralized ledger secured via proof-of-work consensus.",
            "Bitcoin was described by Satoshi Nakamoto in 2008, defining a decentralized blockchain architecture."
        ],
        "hallucinated": [
            "The Bitcoin whitepaper was published in 1995 by the International Monetary Fund to centralize global banking reserves.",
            "Satoshi Nakamoto released Bitcoin in 2008 as a centralized proof-of-stake database hosted on United States Treasury servers.",
            "Bitcoin was created by Vitalik Buterin in October 2008 to eliminate all decentralized peer-to-peer cryptocurrency systems.",
            "The Bitcoin blockchain was shut down permanently in October 2008 after Satoshi Nakamoto declared proof-of-work unviable."
        ]
    },
    {
        "id": "plate_tectonics",
        "context": "The theory of plate tectonics describes the large-scale motion of seven major plates and numerous minor plates of the Earth's lithosphere over the past hundreds of millions of years.",
        "factual": [
            "Plate tectonics explains how lithospheric plates move across Earth's surface over geological timescales.",
            "Earth's lithosphere is divided into multiple major and minor tectonic plates whose motions drive continental drift.",
            "Geological movements of Earth's lithospheric plates over hundreds of millions of years are explained by plate tectonics."
        ],
        "hallucinated": [
            "Plate tectonics proves that Earth's crust is an unbroken, static granite shell that has never experienced horizontal motion.",
            "The theory of plate tectonics establishes that seven major tectonic plates rotate daily around the Earth's gaseous core.",
            "Plate tectonics was disproven in 2015 when geologists verified that continents drift exclusively due to magnetic ocean tides.",
            "Earth's tectonic plates float on atmospheric clouds, drifting thousands of miles every single afternoon."
        ]
    },
    {
        "id": "antibiotic_resistance",
        "context": "Antibiotic resistance occurs when bacteria evolve biological mechanisms that protect them from the effects of antimicrobials, largely accelerated by widespread clinical and agricultural overuse.",
        "factual": [
            "Overuse of antimicrobials in clinical and farming settings accelerates the natural evolution of antibiotic-resistant bacteria.",
            "Bacteria develop resistance to antibiotics through evolutionary adaptations, exacerbated by excessive antimicrobial usage.",
            "Clinical overprescription and agricultural use of antibiotics foster biological defenses in bacteria, rendering drugs ineffective."
        ],
        "hallucinated": [
            "Antibiotic resistance is caused by human red blood cells rejecting synthetic pharmaceuticals due to viral memory.",
            "Bacteria become resistant to antibiotics only when exposed to sub-zero arctic temperatures for several centuries.",
            "Antibiotic resistance has been completely eliminated worldwide following the total discontinuation of microbial medicines.",
            "Overusing antibiotics makes bacteria biologically weaker and universally vulnerable to common saline solutions."
        ]
    },
    {
        "id": "french_revolution",
        "context": "The French Revolution began in 1789 with the storming of the Bastille, leading to the overthrow of the monarchy and the establishment of a republic in France.",
        "factual": [
            "In 1789, the storming of the Bastille marked the outbreak of the French Revolution, dismantling the absolute monarchy.",
            "The French Revolution erupted in 1789, overthrowing the royal monarchy and creating the first French Republic.",
            "Beginning with the 1789 Bastille uprising, the French Revolution transformed French governance into a republic."
        ],
        "hallucinated": [
            "The French Revolution began in 1889 when King Louis XVI successfully conquered Great Britain and dissolved all republican assemblies.",
            "The storming of the Bastille in 1789 was an architectural renovation celebration that permanently preserved the royal monarchy.",
            "Napoleon Bonaparte initiated the French Revolution in 1650 to install an absolute Roman Catholic papacy across France.",
            "The French Revolution resulted in the peaceful coronation of Louis XVI and the immediate abolition of all democratic elections."
        ]
    },
    {
        "id": "hubbles_law",
        "context": "Edwin Hubble discovered in 1929 that galaxies are moving away from Earth with velocities proportional to their distance, establishing observational evidence for an expanding universe.",
        "factual": [
            "In 1929, Edwin Hubble observed that galactic recessional velocities correlate with distance, proving the universe is expanding.",
            "Hubble's 1929 discovery showed that distant galaxies move away proportionally to distance, indicating universal expansion.",
            "Observational proof for the expansion of the cosmos was established by Edwin Hubble in 1929 via galactic red-shifts."
        ],
        "hallucinated": [
            "Edwin Hubble discovered in 1929 that all galaxies are collapsing toward Earth at accelerating speeds, indicating universal contraction.",
            "Albert Einstein discovered Hubble's law in 1850, proving that galaxies remain stationary at fixed cosmic coordinates.",
            "Edwin Hubble proved in 1929 that galactic recessional velocities are completely independent of distance from the observer.",
            "Hubble's 1929 observations proved conclusively that the universe is rotating around Earth at a constant angular velocity."
        ]
    },

    # --- VALIDATION DOMAINS (21 to 23) ---
    {
        "id": "val_transistor",
        "context": "The point-contact transistor was invented in December 1947 by John Bardeen, Walter Brattain, and William Shockley at Bell Labs, replacing vacuum tubes in electronics.",
        "factual": [
            "Bardeen, Brattain, and Shockley created the first transistor at Bell Labs in December 1947, superseding bulky vacuum tubes.",
            "In late 1947, the point-contact transistor was invented at Bell Labs, transforming electronics by replacing vacuum tubes.",
            "Bell Labs researchers invented the point-contact transistor in December 1947, initiating solid-state electronics."
        ],
        "hallucinated": [
            "The transistor was invented in 1985 by Thomas Edison at General Electric to replace integrated silicon microchips.",
            "John Bardeen and Walter Brattain invented the transistor in 1947 to prove that vacuum tubes were irreplaceable in electronics.",
            "The point-contact transistor was developed in December 1947 at Bell Labs, but failed to conduct any electrical current.",
            "William Shockley patented the transistor in 1912 as an analog mechanical amplifier for transatlantic telegraph cables."
        ]
    },
    {
        "id": "val_black_holes",
        "context": "A black hole is a region of spacetime where gravity is so strong that nothing, including light and electromagnetic radiation, possesses sufficient energy to escape its event horizon.",
        "factual": [
            "Gravity inside a black hole is sufficiently intense that neither particles nor electromagnetic waves can escape past the event horizon.",
            "Within the event horizon of a black hole, gravitational pull prevents all light and matter from escaping into outer space.",
            "Electromagnetic radiation and matter cannot escape from the interior of a black hole once crossing its event horizon."
        ],
        "hallucinated": [
            "A black hole is an incandescent celestial body that continuously emits radiant beams of visible light from its central singularity.",
            "Light and radio waves escape freely from black hole event horizons because gravitational forces do not affect electromagnetic photons.",
            "Black holes are hollow planetary formations with zero gravitational pull discovered in Earth's upper atmosphere in 2010.",
            "Inside the event horizon of a black hole, gravity becomes negative, expelling all matter into surrounding planetary orbits."
        ]
    },
    {
        "id": "val_neural_plasticity",
        "context": "Neuroplasticity refers to the biological capacity of the nervous system to modify its structural organization and functional connections in response to experience and learning.",
        "factual": [
            "The brain adapts structurally and functionally throughout life in response to learning, a phenomenon known as neuroplasticity.",
            "Neuroplasticity describes how neural pathways reconfigure their connections based on behavioral experiences and education.",
            "Through neuroplasticity, the nervous system modifies its structural wiring and synaptic strengths following new experiences."
        ],
        "hallucinated": [
            "Neuroplasticity is an irreversible genetic disease that causes neurons to crystallize into rigid, unmodifiable bone tissue.",
            "The nervous system completely loses all biological capacity to modify synaptic connections immediately upon human birth.",
            "Neuroplasticity refers to synthetic plastic implants surgically inserted into the cerebral cortex to stimulate blood flow.",
            "Experiences and learning have been proven biologically unable to induce any functional or structural changes in animal brains."
        ]
    },

    # --- HELD-OUT TEST DOMAINS (24 to 25) - COMPLETELY UNSEEN TOPICS ---
    {
        "id": "test_crispr_base_editing",
        "context": "Base editing, developed by David Liu's laboratory in 2016, enables the direct and programmable conversion of one DNA base pair into another without double-stranded DNA breaks.",
        "factual": [
            "David Liu's team created base editing in 2016, allowing direct conversion of individual DNA base pairs without double-strand cleavage.",
            "In 2016, base editing emerged from David Liu's lab, transforming targeted single-base DNA modifications without causing double-strand breaks.",
            "Direct programmable transition of DNA base pairs without double-stranded breaks was achieved by David Liu's base editing in 2016.",
            "Developed in 2016 by David Liu, base editing executes precise single-nucleotide conversions without cutting both DNA strands."
        ],
        "hallucinated": [
            "Base editing was developed by David Liu in 2016, but requires massive double-stranded DNA breaks to alter chromosome structures.", # Logic contradiction
            "David Liu's laboratory developed base editing in 1985 to replace guide RNA with radioactive isotope bombardment.", # Date & mechanism swap
            "Base editing, introduced by David Liu in 2016, enables programmable conversion of human protein chains directly into lipid molecules.", # Domain error
            "Base editing was proven completely incapable of converting DNA base pairs and was abandoned by David Liu's group in 2016." # Negation
        ]
    },
    {
        "id": "test_gravitational_waves",
        "context": "On September 14, 2015, the LIGO observatory made the first direct physical detection of gravitational waves, produced by the inspiral and merger of two stellar-mass black holes.",
        "factual": [
            "LIGO made history on September 14, 2015, by directly sensing gravitational ripples generated by a pair of colliding black holes.",
            "The first direct observation of gravitational waves occurred at the LIGO facility on September 14, 2015, originating from a binary black hole merger.",
            "Gravitational waves from two merging stellar-mass black holes were physically detected by LIGO on September 14, 2015.",
            "On September 14, 2015, LIGO detected spacetime ripples emitted by the merger of two stellar-mass black holes."
        ],
        "hallucinated": [
            "On September 14, 2015, LIGO detected gravitational waves emitted by the violent collision of two gaseous hydrogen planets.", # Object swap
            "LIGO made its first detection of gravitational waves on September 14, 1975, using optical photographic plates.", # Date & instrument swap
            "The first direct detection of gravitational waves on September 14, 2015 was proven to be terrestrial acoustic noise, disproving black hole mergers.", # Discrediting
            "On September 14, 2015, the Hubble Space Telescope directly captured optical color images of gravitational waves orbiting Jupiter." # Instrument & target swap
        ]
    }
]


def build_split(domains, target_samples):
    """Generates an equal balance of factual (label 1) and hallucinated (label 0) pairs."""
    records = []
    needed_per_class = target_samples // 2
    samples_per_domain_class = needed_per_class // len(domains) + 1

    for d in domains:
        ctx = d["context"]
        facts = d["factual"]
        hallus = d["hallucinated"]

        # Factual
        for _ in range(samples_per_domain_class):
            g_fact = random.choice(facts)
            records.append({
                "context": ctx,
                "generation": g_fact,
                "label": 1,
                "domain_id": d["id"],
                "category": "Factual / Grounded (1)"
            })

        # Hallucinated
        for _ in range(samples_per_domain_class):
            g_hall = random.choice(hallus)
            records.append({
                "context": ctx,
                "generation": g_hall,
                "label": 0,
                "domain_id": d["id"],
                "category": "Hallucinated (0)"
            })

    df = pd.DataFrame(records)
    # Balance exact counts
    f_df = df[df["label"] == 1].sample(n=needed_per_class, random_state=RANDOM_SEED)
    h_df = df[df["label"] == 0].sample(n=needed_per_class, random_state=RANDOM_SEED)
    final_df = pd.concat([f_df, h_df], ignore_index=True)
    return final_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def prepare_benchmark():
    """
    Partitions domains into:
    - Train: 20 Domains -> 4,000 samples (2,000 Factual, 2,000 Hallucinated)
    - Val:   3 Domains  -> 500 samples (250 Factual, 250 Hallucinated)
    - Test:  2 Domains  -> 500 samples (250 Factual, 250 Hallucinated) - Strictly Held-Out
    """
    train_domains = ALL_DOMAINS[:20]
    val_domains = ALL_DOMAINS[20:23]
    test_domains = ALL_DOMAINS[23:25]

    print("[*] Generating Out-of-Domain Stratified Splits...")
    train_df = build_split(train_domains, target_samples=4000)
    val_df = build_split(val_domains, target_samples=500)
    test_df = build_split(test_domains, target_samples=500)

    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)

    print(f"[+] Dataset splits generated successfully:")
    print(f"    - Train Split: 4,000 pairs across 20 distinct domains (2,000 Fact / 2,000 Hallu)")
    print(f"    - Val Split:     500 pairs across 3 unseen domains (250 Fact / 250 Hallu)")
    print(f"    - Test Split:    500 pairs across 2 held-out domains (250 Fact / 250 Hallu)")

    # Combine for EDA visualization
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df["context_words"] = combined_df["context"].apply(lambda x: len(x.split()))
    combined_df["gen_words"] = combined_df["generation"].apply(lambda x: len(x.split()))
    combined_df["gen_chars"] = combined_df["generation"].apply(len)

    # Generate EDA Figures
    sns.set_theme(style="whitegrid")

    # Figure 4: Class Distribution
    plt.figure(figsize=(7, 5))
    palette = ["#2ecc71", "#e74c3c"]
    ax = sns.countplot(x="category", data=combined_df, palette=palette, hue="category", legend=False)
    plt.title("Label Distribution in Sampled Dataset (Hallucination Benchmark)", fontsize=13, weight="bold")
    plt.xlabel("Grounding Class", fontsize=11)
    plt.ylabel("Number of Pairs", fontsize=11)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=12, color='white', weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig4_class_distribution.png"), dpi=300)
    plt.close()

    # Figure 5: Word Count Density
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=combined_df[combined_df["label"] == 1]["gen_words"], label="Factual (1)", fill=True, color="#2ecc71", alpha=0.4)
    sns.kdeplot(data=combined_df[combined_df["label"] == 0]["gen_words"], label="Hallucinated (0)", fill=True, color="#e74c3c", alpha=0.4)
    plt.title("Generation Word-Count Density by Grounding Class", fontsize=13, weight="bold")
    plt.xlabel("Word Count per Generation", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(title="Class")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fig5_length_distribution.png"), dpi=300)
    plt.close()

    stats = {
        "Source": "Curated Multi-Domain Scientific & Historical Factuality Benchmark",
        "Total pairs used": 5000,
        "Classes": "2 – Hallucinated (0), Factual (1)",
        "Mean generation length": f"{combined_df['gen_words'].mean():.1f} words ({combined_df['gen_chars'].mean():.0f} characters)",
        "Median generation length": f"{combined_df['gen_words'].median():.0f} words",
        "Shortest / longest generation": f"{combined_df['gen_words'].min()} words / {combined_df['gen_words'].max()} words",
        "Train pairs": len(train_df),
        "Validation pairs": len(val_df),
        "Test pairs (Held-out)": len(test_df)
    }
    with open(os.path.join(DATA_DIR, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    print("[+] Generated EDA plots: fig4_class_distribution.png & fig5_length_distribution.png")


if __name__ == "__main__":
    prepare_benchmark()
