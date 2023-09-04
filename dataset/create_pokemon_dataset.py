import aiohttp
import requests
import openai
import json
import os
import urllib.request
import asyncio
from aiofiles import open as aio_open
from tqdm import tqdm

# Initialize OpenAI API
openai.api_key = "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"

async def download_image(session, img_url, img_filename):
    try:
        async with session.get(img_url) as resp:
            async with aio_open(img_filename, 'wb') as f:
                await f.write(await resp.read())
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

async def fetch_pokemon(session, url):
    async with session.get(url) as response:
        return await response.json()


async def generate_text(session, names, max_retries=3):
    retries = 0

    while retries < max_retries:
        try:
            messages = [
                {"role": "system", "content": """
                    You are a helpful assistant that generates JSON-formatted descriptions for Pokémon.
                    Don't write too long for each description, or the JSON will be incomplete.
                    For example: 
                    User: Generate descriptions for the following Pokémon: bulbasaur, ivysaur.
                    AI: { 
                        "bulbasaur": "Bulbasaur, the Seed Pokémon. It is a Grass/Poison type. Bulbasaur displays a bright green body, complemented by a large plant bulb growing on its back. This bulb thrives by absorbing sunlight, allowing Bulbasaur to store energy. As this Pokémon evolves, the bulb grows larger and eventually sprouts into a beautiful plant.",
                        "ivysaur": "Ivysaur, the Seed Pokémon and the evolved form of Bulbasaur. It remains a dual-type Grass/Poison Pokémon. Ivysaur possesses a stout body with stronger limbs than its previous form. The bud on its back has now blossomed into a splendid flower. Ivysaur's leaves are toxic, releasing harmful pollen when shaken, causing temporary paralysis on contact.",
                    } 
                    """},
                {"role": "user", "content": f"Generate descriptions for the following Pokémon: {', '.join(names)}."}
            ]

            # You can still specify max tokens and model here
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=500)

            generated_text = response['choices'][0]['message']['content'].strip()

            # Try parsing the JSON to see if it's complete
            parsed_json = json.loads(generated_text)

            return generated_text  # If parsing is successful, return the result

        except json.JSONDecodeError:  # Incomplete JSON
            print(f"Failed to parse JSON (attempt {retries + 1}/{max_retries}). Retrying...")

        except Exception as e:  # Other exceptions like API call failure
            print(f"An error occurred: {e} (attempt {retries + 1}/{max_retries}). Retrying...")

        retries += 1

        if (retries >= max_retries):
            try:
                generated_text = response['choices'][0]['message']['content'].strip() + "\"\n}"

                # Try parsing the JSON to see if it's complete
                parsed_json = json.loads(generated_text)

                return generated_text  # If parsing is successful, return the result
            except:
                print(f"Failed to parse extended JSON.")

    print("Max retries reached. Could not generate text.")


async def main():
    print("Fetching Pokémon data...")
    base_url = "https://pokeapi.co/api/v2/pokemon/"
    offset = 900  # Define your offset variable here
    num_pokemon = 1010

    urls = [f"{base_url}{i}" for i in range(offset + 1, num_pokemon + 1)]

    os.makedirs("train_pokemon_dataset/image", exist_ok=True)

    async with aiohttp.ClientSession() as session:
        print("Fetching Pokémon data...")
        pokemon_list = await asyncio.gather(*(fetch_pokemon(session, url) for url in tqdm(urls)))
        print("Fetched Pokémon data!")

        # Fetch descriptions in batches for efficiency
        batch_size = 5
        annotations = []

        for i in range(0, len(pokemon_list), batch_size):
            print(f"Batch {i // batch_size + 1}/{len(pokemon_list) // batch_size + 1}")
            batch = pokemon_list[i:i + batch_size]
            batch_names = [pokemon['name'] for pokemon in batch]
            print(f"Generating descriptions for {', '.join(batch_names)}...")
            descriptions_text = await generate_text(session, batch_names)
            print("Descriptions generated!")
            print(descriptions_text)
            try:
                descriptions = json.loads(descriptions_text)
                print("Descriptions parsed!")
            except json.JSONDecodeError:
                print("Failed to parse JSON. Skipping this batch.")
                continue

            download_tasks = []
            for pokemon in batch:
                name = pokemon['name']
                img_url = pokemon['sprites']['other']['official-artwork']['front_default']
                img_filename = f"train_pokemon_dataset/image/{pokemon['id']}.png"
                annotations.append({
                    "image_id": str(pokemon["id"]),
                    "caption": descriptions.get(name, f"Description for {name}")
                })

                download_tasks.append(download_image(session, img_url, img_filename))

            # Download all images in the batch simultaneously
            print("Downloading images...")
            await asyncio.gather(*download_tasks)
            print("Images downloaded!")

            # Save after each batch
            print("Saving annotations...")
            with open("train_pokemon_dataset/filter_cap.json", 'w') as f:
                json.dump({"annotations": annotations}, f, indent=4)
            print("Annotations saved!")

if __name__ == '__main__':
    asyncio.run(main())