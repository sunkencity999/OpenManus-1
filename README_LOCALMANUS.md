# LocalManus

## Overview

LocalManus is a fork of the OpenManus project with enhanced reasoning capabilities, task completion, and content generation features. It's designed to provide a more reliable and effective agent experience with minimal redundant questioning and guaranteed task completion.

## Key Features

### 1. Enhanced Task Completion
- Automatically completes tasks after a reasonable number of steps
- Generates structured deliverables for various task types
- Supports specialized content creation (marketing plans, poems, research reports)

### 2. Improved Web Navigation
- Better URL detection and handling
- Support for URL fragments (#section) navigation
- Enhanced content extraction from websites

### 3. Reduced Redundant Questioning
- Tracks previous questions to avoid repetition
- Maintains context across interactions
- Makes better decisions about when user input is truly needed

### 4. File Handling
- Automatically detects file paths in task descriptions
- Saves content to requested files
- Provides proper error handling for file operations

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

### Example Usage

```
$ python main.py
Enter your task: Create a poem about Harriet Tubman in the style of Stephen King
```

## Task Types Supported

- **Marketing Plans**: Target audience, value proposition, marketing channels, etc.
- **Business Plans**: Executive summary, company description, market analysis, etc.
- **Poems**: Various styles including Stephen King, Shakespeare, and haiku
- **Research Reports**: Overview, key findings, conclusion

## Development Roadmap

1. Add more task types and templates
2. Improve browser navigation capabilities
3. Enhance information extraction from websites
4. Add more specialized content creation capabilities
5. Implement better memory management for long-term context

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the same terms as the original OpenManus project.

## Acknowledgments

- Based on the OpenManus project
- Improved with enhanced reasoning capabilities and task completion logic
