
def process_triples(triples):
    """
    Robustly processes the figurative reasoning string to extract relevant sections.
    Specifically designed to handle structured reasoning with nested sections.
    """
    if not isinstance(triples, str):
        return "Missing reasoning"

    # Normalize the text to handle different formatting styles
    triples = triples.replace('**Cause-effect:**', '**Cause-effect**')
    triples = triples.replace('**Cause-effect :**', '**Cause-effect**')

    # Define regex patterns to extract sections
    section_patterns = [
        {
            'name': 'cause-effect',
            'pattern': r'\*\*Cause-effect\*\*\s*:?\s*(.+?)(?=\*\*Figurative|$)',
            'clean_pattern': r'\s*-\s*'
        },
        {
            'name': 'figurative understanding',
            'pattern': r'\*\*Figurative Understanding\*\*\s*:?\s*(.+?)(?=\*\*Mental|$)',
            'clean_pattern': r'\s*-\s*'
        },
        {
            'name': 'mental state',
            'pattern': r'\*\*Mental State\*\*\s*:?\s*(.+?)(?=$|These triples|$)',
            'clean_pattern': r'\s*-\s*'
        }
    ]

    # Function to clean and extract content
    def extract_section_content(section_text, clean_pattern):
        # Remove markdown formatting
        section_text = re.sub(r'\*\*[^*]+\*\*', '', section_text)
        
        # Split into individual points
        points = [p.strip() for p in re.split(clean_pattern, section_text) if p.strip()]
        
        # Clean each point
        cleaned_points = []
        for point in points:
            # Remove parentheses if present
            point = re.sub(r'^\s*\(|\)\s*$', '', point)
            # Remove extra whitespace
            point = re.sub(r'\s+', ' ', point).strip()
            if point:
                cleaned_points.append(point)
        
        return ' '.join(cleaned_points) if cleaned_points else ''

    # Extract sections
    extracted_sections = {}
    for section in section_patterns:
        match = re.search(section['pattern'], triples, re.DOTALL | re.IGNORECASE)
        if match:
            content = extract_section_content(match.group(1), section['clean_pattern'])
            if content:
                extracted_sections[section['name']] = content

    # Construct output
    if extracted_sections:
        output_lines = []
        for section, content in extracted_sections.items():
            output_lines.append(f"{section.lower()}: {content}")
        return '\n'.join(output_lines)
    
    # Fallback if no sections found
    return "Reasoning sections not identified"
# =========================================================
