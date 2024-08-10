import pandas as pd

# Sample data for SDG 9: Industry, Innovation, and Infrastructure
data = {
    'text': [
        "The development of smart cities is a crucial step towards sustainable infrastructure.",
        "Innovations in renewable energy technologies are driving industrial growth.",
        "Investments in transportation infrastructure are essential for economic development.",
        "The rise of automation and AI is transforming industries worldwide.",
        "Building resilient infrastructure is key to reducing economic and environmental vulnerabilities.",
        "The history of the steam engine and its impact on industrialization.",
        "Developing countries are focusing on enhancing their manufacturing capabilities.",
        "Introduction of 5G technology is set to revolutionize communication infrastructure.",
        "The importance of industrial clusters in promoting innovation and competitiveness.",
        "Non-SDG topic: Exploring the culinary traditions of France.",
        "Non-SDG topic: The impact of climate change on polar bear populations.",
        "Innovative financing models are needed to support large-scale infrastructure projects.",
        "The role of public-private partnerships in infrastructure development.",
        "Non-SDG topic: A guide to the best hiking trails in the Rocky Mountains.",
        "The impact of digital transformation on the manufacturing sector.",
        "Resilient infrastructure is crucial for disaster risk reduction.",
        "Non-SDG topic: The effects of fast fashion on global markets.",
        "Sustainable industrial practices are necessary for long-term economic growth.",
        "Investment in research and development is driving technological innovation.",
        "The expansion of broadband networks is vital for connecting remote areas."
    ],
    'label': [
        'Infrastructure Development',
        'Innovation',
        'Infrastructure Development',
        'Industrial Development',
        'Infrastructure Development',
        'Industrial Development',
        'Industrial Development',
        'Infrastructure Development',
        'Innovation',
        'Non-SDG Aligned',
        'Non-SDG Aligned',
        'Infrastructure Development',
        'Infrastructure Development',
        'Non-SDG Aligned',
        'Industrial Development',
        'Infrastructure Development',
        'Non-SDG Aligned',
        'Industrial Development',
        'Innovation',
        'Infrastructure Development'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('data_Q2.csv', index=False)

print("SDG 9 dataset created and saved as 'sdg9_data.csv'.")