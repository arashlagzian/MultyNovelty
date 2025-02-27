import os
import random
import numpy as np
import nltk
import json
from nltk.tokenize import word_tokenize
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import argparse
import torchvision 
# Ensure NLTK packages are downloaded
# nltk.download('punkt')
#print(torch.__version__, torchvision.__version__)

set_seed(42)
random.seed(42)
np.random.seed(42)


os.environ["CUDA_VISIBLE_DEVICES"] = "5"  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)


prompts = [
    # 1.Philosophical Question:
    "What is the meaning of true happiness in life?",

    # 2.Hypothetical Scenario:
    "If humans could live on Mars, what challenges would they face and how could they overcome them?",

    # 3.Creative Thinking Prompt:
    "Can you describe an imaginary city where technology and nature exist in perfect harmony?",

    # 4.Practical Advice Question:
    "What are the most effective ways to learn a new language quickly?",

    # 5.Exploration of Abstract Concepts:
    "How would you explain the concept of time to someone who has never experienced it?",
    
    # 6.Scientific Exploration:
    "What are the possible effects of artificial intelligence on scientific research in the next decade?",

    # 7.Ethical Dilemma:
    "Is it ever justifiable to prioritize technological advancement over environmental protection?",

    # 8.Problem-Solving Question:
    "How can cities effectively reduce traffic congestion without compromising accessibility?",

    # 9.Imaginative Scenario:
    "If animals could communicate with humans, how would that change our world?",

    # 10.Personal Reflection Prompt:
    "What qualities make someone a great leader, and how can those qualities be developed?"
    ]



text_views = {
    # 1. Philosophical Question
    "What is the meaning of true happiness in life?": [
        "Describe true happiness from a scientific perspective.",
        "Write a poem about true happiness.",
        "Explain true happiness through ancient philosophy.",
        "Tell a short story about someone discovering happiness.",
        "List key features of true happiness.",
        "Create a dialogue between two friends discussing happiness.",
        "Explain the psychology behind true happiness.",
        "Reflect philosophically on true happiness.",
        "Write the history of happiness over time.",
        "Write a news article about a discovery related to happiness.",
        "List fun facts about happiness research.",
        "Compare happiness to material wealth.",
        "Imagine a dystopia where happiness is forbidden.",
        "Describe happiness as a recipe.",
        "Explain happiness as a mathematical formula.",
        "Write a bedtime story about happiness.",
        "Analyze happiness in literature.",
        "Describe happiness through an animal’s eyes.",
        "Create a myth about the origin of happiness.",
        "Write a letter to your future self about happiness.",
        "Debate whether happiness is a destination or journey.",
        "Imagine happiness as a person and describe them.",
        "Explain happiness using natural metaphors.",
        "Frame happiness as a civilization-building tool.",
        "Predict the future of happiness.",
        "Analyze cultural differences in happiness.",
        "Describe happiness as part of a dream.",
        "Create a fairy tale about finding happiness.",
        "Write a motivational speech about happiness.",
        "Reflect on technology’s effects on happiness.",
        "Compose song lyrics about happiness.",
        "Describe a utopia where happiness is guaranteed.",
        "Imagine a science experiment measuring happiness.",
        "Write a journal entry about true happiness.",
        "Describe happiness as art.",
        "Frame happiness as the solution to a crisis.",
        "Write about happiness as a treasure.",
        "Create a step-by-step happiness guide.",
        "Analyze happiness through neuroscience.",
        "Imagine programmable happiness in the future.",
        "Discuss happiness as a social construct.",
        "Imagine a courtroom debate about happiness.",
        "Explore happiness as evolving emotion.",
        "Describe happiness as energy.",
        "Interview 'happiness' as a character.",
        "Write about happiness as survival.",
        "Design a quiz to measure happiness.",
        "Discuss ethics of artificial happiness.",
        "Frame happiness as a puzzle.",
        "Explore happiness tied to memory.",
        "Create a fantasy where happiness grants powers.",
        "Write a eulogy for a happy life."
    ],

    # 2. Hypothetical Scenario
    "If humans could live on Mars, what challenges would they face and how could they overcome them?": [
        "Describe the daily life of a Mars colonist.",
        "Explain the technology needed for survival.",
        "Write a poem about settling on Mars.",
        "Imagine a diary entry from a Martian settler.",
        "List survival strategies for Mars.",
        "Create a dialogue between colonists debating terraforming.",
        "Reflect on the psychological impact of isolation.",
        "Discuss ethical issues of colonizing Mars.",
        "Write a futuristic news article about Mars development.",
        "Describe a Mars-based economy.",
        "Analyze the effects of low gravity.",
        "Imagine cultural traditions forming on Mars.",
        "Predict political structures on Mars.",
        "Write about Mars-inspired art movements.",
        "Explain how agriculture could work.",
        "Design a habitat for Mars settlers.",
        "Explore mining as an industry on Mars.",
        "Discuss education in Martian colonies.",
        "Write about a conflict between Earth and Mars.",
        "Imagine alien contact on Mars.",
        "Describe the first wedding on Mars.",
        "Create a survival manual.",
        "Analyze Mars travel costs.",
        "Write a sci-fi short story.",
        "Describe religion evolving on Mars.",
        "Reflect on humanity’s legacy in space.",
        "Debate resource ownership rights.",
        "Discuss ecological effects of terraforming.",
        "Analyze mental health strategies.",
        "Write about medical advancements for Mars.",
        "Describe cultural holidays on Mars.",
        "Create marketing slogans for Mars tourism.",
        "Design educational programs for Mars children.",
        "Explore relationships in isolation.",
        "Write about Mars-inspired inventions.",
        "Imagine timekeeping adaptations.",
        "Discuss adapting Earth animals for Mars.",
        "Describe a Mars-based sport.",
        "Analyze the effects of Martian dust storms.",
        "Explain food production challenges.",
        "Write about fashion trends for Mars.",
        "Create a guide for exploring Mars.",
        "Write a philosophical essay on leaving Earth.",
        "Debate the ethics of Mars colonization.",
        "Create a fiction involving crime on Mars.",
        "Discuss Mars settlement governance models.",
        "Design Martian leisure activities.",
        "Write an obituary for the first Martian settler.",
        "Analyze cross-cultural relations on Mars.",
        "Imagine Mars tourism brochures.",
        "Write about a failed Mars mission."
    ],
    
    # 3. Creative Thinking Prompt
    "Can you describe an imaginary city where technology and nature exist in perfect harmony?": [
        "Describe the architecture of the city.",
        "Write a poem about the beauty of the city.",
        "Create a short story about a visitor exploring the city.",
        "List the technological features that enable harmony.",
        "Create a dialogue between two citizens discussing life in the city.",
        "Reflect philosophically on the balance between nature and technology.",
        "Write a news article about a groundbreaking invention in the city.",
        "Describe a day in the life of a resident.",
        "Imagine an art festival held in the city.",
        "Write about transportation systems in the city.",
        "Explain how waste management is handled.",
        "Design an education system for the city.",
        "Create a legend about the city's origins.",
        "Discuss the role of artificial intelligence in governance.",
        "Describe the flora and fauna coexisting with humans.",
        "Imagine a tourist guide to attractions in the city.",
        "Write a recipe book inspired by the city’s fusion culture.",
        "Discuss laws that protect both nature and technology.",
        "Analyze economic systems that sustain harmony.",
        "Write about community rituals and festivals.",
        "Describe how energy is produced and consumed.",
        "Create a survival plan in case of disaster.",
        "Write about how the city prepares for the future.",
        "Explore the spiritual practices of the inhabitants.",
        "Analyze how cultural values promote harmony.",
        "Describe celebrations of environmental milestones.",
        "Create a list of futuristic occupations in the city.",
        "Write about the role of music and art.",
        "Explain how healthcare systems function.",
        "Create a guidebook for visitors.",
        "Discuss the city’s history and evolution.",
        "Design a system of sustainable agriculture.",
        "Analyze the impact of climate on architecture.",
        "Write about water conservation methods.",
        "Describe underground structures or hidden systems.",
        "Write a bedtime story for children about the city.",
        "Imagine a protest movement in the city.",
        "Explain how the city handles migration and growth.",
        "Reflect on how harmony affects mental health.",
        "Write about sports and recreation.",
        "Discuss the role of ethics in urban planning.",
        "Explore innovations that allow wildlife to thrive.",
        "Describe public spaces and their cultural significance.",
        "Write about the city’s role as a global leader.",
        "Create a fictional interview with the city’s architect.",
        "Imagine the city in 100 years.",
        "Design safety systems for emergencies.",
        "Write about technological art installations.",
        "Describe nighttime activities and entertainment.",
        "Write about ancient ruins hidden in the city.",
        "Analyze the cultural impact of technology in the city."
    ],

    # 4. Practical Advice Question
    "What are the most effective ways to learn a new language quickly?": [
        "List tips and tricks for quick language learning.",
        "Write a motivational speech for language learners.",
        "Create a daily schedule for language practice.",
        "Describe the role of technology in language acquisition.",
        "Write a short story about someone mastering a language.",
        "Explain the neuroscience behind learning languages.",
        "Provide a case study of a successful language learner.",
        "Design a language-learning mobile app concept.",
        "Write a dialogue between a teacher and student.",
        "List fun language games for faster learning.",
        "Analyze the benefits of immersion programs.",
        "Explain how cultural exposure aids language learning.",
        "Create a step-by-step guide for beginners.",
        "Write a poem about language as a bridge.",
        "Describe the role of music in language learning.",
        "Imagine a futuristic device that teaches languages instantly.",
        "Write a letter from someone learning a new language.",
        "Discuss common challenges and how to overcome them.",
        "Reflect on the psychology of motivation in learning.",
        "Analyze the impact of social media on language acquisition.",
        "Describe an ideal language-learning classroom.",
        "List online resources and platforms for learning languages.",
        "Create a fictional language-learning competition.",
        "Discuss the ethics of AI translation tools.",
        "Analyze the effect of body language on communication.",
        "Write about language learning as a lifelong skill.",
        "Describe a travel experience enhanced by language skills.",
        "Imagine a language-learning robot companion.",
        "Create a debate about traditional vs. modern learning methods.",
        "List strategies for memorizing vocabulary.",
        "Discuss the influence of multilingualism on brain development.",
        "Analyze the benefits of learning multiple languages.",
        "Design a workbook template for language practice.",
        "Describe storytelling as a tool for learning languages.",
        "Write about the role of humor in language acquisition.",
        "Create a checklist for effective language study habits.",
        "Discuss cultural sensitivity and its role in learning languages.",
        "Analyze language-learning myths and debunk them.",
        "Write about maintaining fluency after learning.",
        "Describe how accents develop and their importance.",
        "Imagine a reality show about language learners.",
        "Explore how children vs. adults learn languages differently.",
        "Write about personal growth through language learning.",
        "Create a fictional support group for learners.",
        "Write about learning through mistakes and humor.",
        "Discuss language learning in virtual reality.",
        "Analyze the role of writing and reading in mastery.",
        "Describe dream interpretation in learning languages.",
        "Write about using movies and TV shows to learn.",
        "Create a journal template for tracking progress.",
        "Imagine a futuristic translation implant technology."
    ],

    # 5. Exploration of Abstract Concepts
    "How would you explain the concept of time to someone who has never experienced it?": [
        "Describe time as a flowing river.",
        "Explain time using cycles in nature, like seasons.",
        "Compare time to a spiral that repeats but progresses.",
        "Write a poem capturing the essence of time.",
        "Create a fable where time is a character.",
        "Describe time as a heartbeat or rhythm.",
        "Explain time as a fourth dimension in physics.",
        "Use metaphors of sand slipping through an hourglass.",
        "Write a short story about someone discovering time.",
        "Imagine a world where time doesn’t exist.",
        "Compare time to light and shadows.",
        "Explain time as a series of moments frozen in frames.",
        "Describe time as an endless loop or a straight line.",
        "Reflect philosophically on time’s role in life.",
        "Discuss time as a tool for measuring change.",
        "Write about time as an illusion or construct.",
        "Explain time using astronomical events.",
        "Imagine time as a currency used for transactions.",
        "Describe how time shapes memories and emotions.",
        "Write a letter to someone from the past about time.",
        "Reflect on time as an experience rather than a measure.",
        "Write about time in dreams vs. reality.",
        "Create a dialogue between a child and a scientist about time.",
        "Describe time as a song with verses and chorus.",
        "Write about time stopping and restarting.",
        "Imagine time as an infinite library of events.",
        "Discuss cultural differences in perceiving time.",
        "Describe time as layers of history.",
        "Analyze time through the lens of relativity.",
        "Write about time as the glue that holds reality together.",
        "Explain time through the life cycle of a butterfly.",
        "Write a futuristic story where time can be paused.",
        "Describe time as a flowing river of energy.",
        "Analyze time’s role in technology and innovation.",
        "Create an art installation concept to visualize time.",
        "Describe time using light-speed travel analogies.",
        "Write about time from the perspective of a clock.",
        "Discuss the connection between time and entropy.",
        "Write about time dilation and space travel.",
        "Imagine time as a garden growing different flowers.",
        "Describe time as a sculptor shaping the present.",
        "Write about timelessness and eternity.",
        "Compare time to tides controlled by unseen forces.",
        "Create a myth about the origin of time.",
        "Explain time as a loop in quantum theory.",
        "Reflect on living without the concept of time.",
        "Write about nostalgia as fragments of time.",
        "Analyze language’s role in shaping time perception.",
        "Imagine conversations between past, present, and future.",
        "Describe time as a mosaic of fragmented events."
    ],

    # 6. Scientific Exploration
    "What are the possible effects of artificial intelligence on scientific research in the next decade?": [
        "Analyze how AI accelerates data processing.",
        "Discuss AI’s role in discovering new drugs.",
        "Write a news article about AI breakthroughs.",
        "Describe AI’s impact on climate modeling.",
        "Explain how AI automates repetitive research tasks.",
        "Reflect philosophically on AI replacing human intuition.",
        "Imagine AI collaborating with scientists in space exploration.",
        "Discuss ethical concerns about AI in research.",
        "Write a fictional dialogue between a scientist and an AI.",
        "Create a timeline of AI advancements in research.",
        "List AI’s advantages and limitations in data analysis.",
        "Write about AI creating new mathematical models.",
        "Discuss AI’s role in simulations and virtual experiments.",
        "Describe AI helping decode human DNA sequences.",
        "Analyze AI’s influence on materials science discoveries.",
        "Predict AI’s role in designing sustainable energy solutions.",
        "Write a futuristic story about AI-powered research labs.",
        "Explain how AI might redefine scientific creativity.",
        "Analyze AI’s role in improving robotics for experiments.",
        "Discuss AI’s application in astronomy and space exploration.",
        "Write about AI helping predict natural disasters.",
        "Examine AI’s potential to detect patterns in big data.",
        "Describe AI acting as virtual teachers for scientists.",
        "Imagine AI collaborating with biologists to study evolution.",
        "Analyze how AI optimizes experiments for efficiency.",
        "Write about AI’s role in creating personalized medicine.",
        "Explain AI’s use in uncovering ancient languages or artifacts.",
        "Debate AI’s ability to innovate vs. follow patterns.",
        "Discuss the risks of over-reliance on AI in science.",
        "Describe AI models that predict pandemics.",
        "Imagine AI writing and reviewing scientific papers.",
        "Reflect on AI as a partner versus a competitor.",
        "Explain how AI aids in visualizing molecular structures.",
        "Describe AI’s impact on quantum computing advancements.",
        "Write about AI-controlled robots exploring deep-sea mysteries.",
        "Analyze AI’s role in advancing nanotechnology.",
        "Explain AI’s function in automating genome editing.",
        "Discuss AI’s impact on renewable energy research.",
        "Write about AI helping simulate black hole formation.",
        "Imagine AI systems that learn directly from nature.",
        "Discuss AI’s role in detecting climate patterns.",
        "Write about AI solving complex mathematical puzzles.",
        "Describe how AI monitors ecosystems and wildlife patterns.",
        "Analyze AI’s contribution to bioinformatics.",
        "Reflect on AI as a source of inspiration for humans.",
        "Imagine AI-powered citizen science initiatives.",
        "Explain AI’s role in speeding up clinical trials.",
        "Discuss AI’s influence on the study of consciousness.",
        "Write about AI in monitoring and preventing cyber threats.",
        "Create a fictional AI assistant guiding human scientists."
    ],

    # 7. Ethical Dilemma
    "Is it ever justifiable to prioritize technological advancement over environmental protection?": [
        "Write a debate between two scientists on this issue.",
        "Analyze the ethical implications of prioritizing technology.",
        "Describe a futuristic world where technology triumphed over nature.",
        "Imagine a dystopia caused by ignoring environmental protection.",
        "List the pros and cons of prioritizing technology.",
        "Write a news article about a controversial technology project.",
        "Reflect philosophically on humanity’s role as caretakers of Earth.",
        "Create a poem highlighting the conflict between technology and nature.",
        "Write a dialogue between a tech developer and an environmentalist.",
        "Imagine an AI tasked with deciding between nature and innovation.",
        "Discuss the moral responsibilities of tech companies.",
        "Analyze how regulations balance progress and sustainability.",
        "Create a short story about restoring nature using advanced technology.",
        "Write about a society living entirely underground to protect the planet.",
        "Debate whether sustainable technology can resolve the conflict.",
        "Describe nature reclaiming urban spaces after technological collapse.",
        "Explain how technology could mimic natural processes.",
        "Discuss carbon credits as a tool for balancing development.",
        "Write a letter to future generations about preserving nature.",
        "Analyze historical examples of industrialization vs. nature.",
        "Describe a utopia where technology and nature coexist peacefully.",
        "Examine cultural perspectives on valuing nature over progress.",
        "Imagine a simulation to test environmental sustainability.",
        "Write about technology enabling eco-restoration projects.",
        "Discuss bioengineering solutions to environmental challenges.",
        "Explore renewable energy as a compromise between tech and nature.",
        "Write about species engineered to thrive in polluted environments.",
        "Analyze the economic impacts of prioritizing nature.",
        "Imagine a world where AI governs environmental policies.",
        "Debate banning certain technologies to protect ecosystems.",
        "Write about the psychological effects of living in high-tech cities.",
        "Examine how environmental disasters influence technological choices.",
        "Describe the role of education in balancing technology and nature.",
        "Create a myth about humans losing their connection with nature.",
        "Write about scientists inventing artificial forests.",
        "Reflect on nature as inspiration for biomimetic technologies.",
        "Analyze the effectiveness of green technologies.",
        "Imagine Earth after 100 years of unchecked technological growth.",
        "Create a fictional treaty to balance innovation and sustainability.",
        "Debate how climate change affects technological development.",
        "Write about the ethics of mining resources from other planets.",
        "Discuss cultural rituals that emphasize nature preservation.",
        "Explain how indigenous knowledge informs modern sustainability.",
        "Write about water scarcity in a world dominated by technology.",
        "Analyze trends in eco-friendly startups and innovations.",
        "Discuss geoengineering as a tool for climate repair.",
        "Create a fictional protest against technological expansion.",
        "Reflect on humanity’s addiction to convenience over sustainability.",
        "Write about wildlife adapting to urbanized environments.",
        "Examine the ethical concerns of genetic engineering in ecosystems."
    ],

    # 8. Problem-Solving Question
    "How can cities effectively reduce traffic congestion without compromising accessibility?": [
        "List smart traffic management technologies.",
        "Analyze the impact of public transportation expansion.",
        "Write a futuristic story about flying cars replacing roads.",
        "Describe how AI can optimize traffic signals.",
        "Propose urban planning strategies to improve walkability.",
        "Write a news article about a successful traffic reduction project.",
        "Create a dialogue between a mayor and a city planner.",
        "Describe a city where all transportation is underground.",
        "Analyze the effects of carpooling incentives.",
        "Imagine a world without personal vehicles.",
        "List features of an ideal traffic-free city.",
        "Describe autonomous vehicles coordinating traffic flow.",
        "Propose policies for promoting biking and walking.",
        "Write a letter to citizens about a new traffic solution.",
        "Examine congestion pricing as a method to reduce traffic.",
        "Create a public awareness campaign script.",
        "Discuss remote work's role in reducing traffic.",
        "Write a debate about banning cars in city centers.",
        "Explain the role of drones and delivery robots.",
        "Design a city layout optimized for pedestrians.",
        "Analyze how zoning laws affect traffic patterns.",
        "Imagine a mobile app for carpooling in real time.",
        "Describe the impact of telecommunication infrastructure.",
        "Write about hyperloop systems as a transportation revolution.",
        "Reflect on cultural shifts required for less car dependency.",
        "Propose a rewards system for using public transit.",
        "Analyze the impact of ridesharing platforms.",
        "Discuss urban agriculture reducing the need for transport.",
        "Create a sci-fi story about teleportation solving traffic.",
        "Imagine aerial tramways connecting neighborhoods.",
        "Propose parking solutions to reduce road congestion.",
        "Analyze bike-sharing programs and their effectiveness.",
        "Discuss underground highway tunnels as alternatives.",
        "Explain the effects of high-speed rail systems.",
        "Write about self-driving buses optimizing city transport.",
        "Imagine green corridors replacing major roads.",
        "Examine electric scooters and micro-mobility devices.",
        "Write a motivational speech promoting public transport.",
        "Describe the benefits of flexible work schedules.",
        "Analyze pedestrian bridges and skywalks as solutions.",
        "Create a fictional city with vertical transportation.",
        "Reflect on nature-integrated transport solutions.",
        "Write about traffic congestion in megacities.",
        "Propose sensors that monitor and adapt traffic flow.",
        "Analyze public opinion on congestion charges.",
        "Create a campaign advocating for smarter urban planning.",
        "Describe neighborhood-based vehicle rental systems.",
        "Imagine AI traffic controllers managing all vehicles.",
        "Reflect on reducing personal vehicle ownership.",
        "Propose a floating transportation network on water bodies.",
        "Discuss futuristic urban cable car systems."
    ],

    # 9. Imaginative Scenario
    "If animals could communicate with humans, how would that change our world?": [
        "Write a short story about a day when animals started talking.",
        "Create a dialogue between a human and their pet.",
        "Imagine a courtroom trial where an animal testifies.",
        "Analyze the ethical implications of eating animals.",
        "Describe laws protecting animal rights in this world.",
        "Write a poem reflecting on the wisdom of animals.",
        "Imagine animals forming their own political systems.",
        "Write a news article about the first human-animal treaty.",
        "Reflect philosophically on the nature of intelligence.",
        "Describe how zoos would transform in this world.",
        "Write about schools teaching animal languages.",
        "Imagine a debate between a cat and a dog.",
        "Discuss how wildlife conservation efforts might change.",
        "Describe a protest organized by animals.",
        "Write a futuristic documentary script about animal societies.",
        "Explain how interspecies friendships could evolve.",
        "Imagine humans learning survival skills from animals.",
        "Describe therapy animals offering emotional advice.",
        "Write about the role of animals in politics.",
        "Imagine animals acting as mediators in human conflicts.",
        "Write a bedtime story about a talking forest.",
        "Reflect on how religions might change in this world.",
        "Create a fable featuring an animal kingdom.",
        "Describe business partnerships between humans and animals.",
        "Write a comedy sketch about animals negotiating with humans.",
        "Analyze economic impacts of hiring animals for labor.",
        "Create laws protecting animal privacy.",
        "Write about space exploration with animal astronauts.",
        "Discuss language barriers even among animal species.",
        "Create advertisements featuring animals speaking for brands.",
        "Imagine animals giving lectures at universities.",
        "Describe a therapy session with a wise elephant.",
        "Write about animals as judges in legal disputes.",
        "Reflect on animals becoming celebrities and influencers.",
        "Imagine spies using animals for undercover operations.",
        "Discuss conflicts between herbivores and carnivores.",
        "Create a religious movement inspired by animal wisdom.",
        "Write a musical featuring animals as performers.",
        "Describe how agriculture would change with speaking animals.",
        "Analyze the impact on pet ownership and breeding.",
        "Imagine rescue missions led by animal teams.",
        "Write about historical events retold by animals.",
        "Explore cross-species friendships in schools.",
        "Describe therapy programs led by dolphins.",
        "Analyze climate activism by speaking animals.",
        "Write about animals negotiating natural resource use.",
        "Imagine a wildlife census conducted by animals.",
        "Describe futuristic technology enabling human-animal translation.",
        "Reflect on philosophical discussions with ancient tortoises.",
        "Write about the rise of animal-led revolutions.",
        "Create a documentary about the secret lives of animals."
    ],

    # 10. Personal Reflection Prompt
    "What qualities make someone a great leader, and how can those qualities be developed?": [
        "List key traits of great leaders and their meanings.",
        "Write a motivational speech about leadership.",
        "Analyze leadership lessons from historical figures.",
        "Describe how adversity builds leadership qualities.",
        "Write a poem about inspiring leadership.",
        "Create a fictional leader who embodies greatness.",
        "Reflect philosophically on the nature of leadership.",
        "Discuss whether leadership is innate or learned.",
        "Imagine leadership training in futuristic societies.",
        "Write a diary entry from a leader facing challenges.",
        "Analyze cultural differences in leadership styles.",
        "Write a short story about a leader’s transformation.",
        "Describe leadership lessons learned through failure.",
        "Create a debate about authoritarian vs. democratic leadership.",
        "Reflect on mentorship as a path to leadership development.",
        "Analyze leadership through the lens of psychology.",
        "Write a fable about leadership and responsibility.",
        "Discuss ethical dilemmas faced by leaders.",
        "Describe the importance of empathy in leadership.",
        "Imagine AI-powered leadership coaches.",
        "Write about how leaders inspire change.",
        "List habits that foster effective leadership.",
        "Reflect on the role of vulnerability in leadership.",
        "Analyze how communication defines great leadership.",
        "Write about cultural rituals for honoring leaders.",
        "Discuss the impact of failures in shaping leaders.",
        "Imagine a school dedicated to leadership training.",
        "Create a fictional interview with an inspiring leader.",
        "Describe the balance between power and humility.",
        "Write about leaders who changed history silently.",
        "Analyze leadership in nature, like animal hierarchies.",
        "Describe the influence of storytelling in leadership.",
        "Write a letter to a young aspiring leader.",
        "Imagine leadership training in virtual reality.",
        "Discuss the importance of adaptability in leadership.",
        "Analyze the role of integrity in building trust.",
        "Write about leaders navigating moral dilemmas.",
        "Create a dialogue between past and future leaders.",
        "List books and films about great leadership.",
        "Write about cultural myths defining leadership.",
        "Imagine community-driven leadership models.",
        "Analyze leadership styles in different industries.",
        "Reflect on team-building as a leadership skill.",
        "Write about historical protests led by iconic leaders.",
        "Describe qualities that make leaders memorable.",
        "Imagine AI systems guiding leadership development.",
        "Analyze how failures teach resilience to leaders.",
        "Write about the role of creativity in leadership.",
        "Create a futuristic society led by rotating leadership roles.",
        "Discuss how modern technology reshapes leadership styles.",
        "Describe leadership as a journey, not a destination."
    ]    
}

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


def build_prompt(user_content):
    system_message = (
                            "You are a helpful assistant.Your answers should be plain text "
                            "without using special characters, bullet points, or lists. "
                            "Don't repeat the question in your answer and just provide the answer "
                            "at most 125 tokens."
                        )
    prompt_text = ( 
        f"System: {system_message}\n"
        f"User: {user_content}\n"
        "Assistant:"
    )
    return prompt_text


def generate_outputs(prompt, views, num_samples, batch_size, max_length, temperature=0.9):
    outputs = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        
        selected_views = random.choices(views, k=min(batch_size, num_samples - len(outputs)))

        
        batch_prompts = [f"{prompt} {view}" for view in selected_views]

        
        prompt_texts = [build_prompt(p) for p in batch_prompts]

        
        model_inputs = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        ).to(model.device)

        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.95
        )

       
        for input_ids, output_ids, view, full_prompt_text in zip(
            model_inputs["input_ids"],
            generated_ids,
            selected_views,
            prompt_texts
        ):
            generated_ids_trimmed = output_ids[len(input_ids):]
            response = tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True)

            # ---------------------------
            # LOGGING
            # ---------------------------
            #print("=" * 70)
            #print(f"** Batch Index: {batch_idx + 1}")
            #print(f"Randomly Selected View: {view}")
            #print(f"Full Prompt Text:")
            #print(full_prompt_text)
            #print(f"Assistant Response:")
            #print(response)
            #print("=" * 70)

            outputs.append(response)

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text outputs from prompts.")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to generate per prompt.")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum length of generated text.")
    args = parser.parse_args()

    num_samples = args.num_samples
    max_length = args.max_length
    batch_size = 100  

    print(f"Generating {num_samples} samples with max length {max_length}...")

    
    all_outputs = {}
    for prompt in prompts:
        outputs = generate_outputs(
            prompt,
            views=text_views[prompt],
            num_samples=num_samples,
            batch_size=batch_size,
            max_length=max_length
        )
        all_outputs[prompt] = outputs

    
    output_file = f"deepseek_tv_{num_samples}_{max_length}.json"
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=4)

    print(f"Outputs saved to {output_file}")

    # Print sample
    #for prompt, generated_texts in all_outputs.items():
    #    print(f"Prompt: {prompt}")
    #    for i, text in enumerate(random.sample(generated_texts, min(3, len(generated_texts)))):
    #        print(f"Output {i+1}: {text}")
    #    print("-" * 80)