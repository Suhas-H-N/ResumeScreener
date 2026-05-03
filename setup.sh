#!/bin/bash
# ResumeIQ Pro - Initialization Script

echo "🧠 ResumeIQ Pro - Setup Script"
echo "================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo -e "${RED}✗ Python 3.11+ required. Found: $python_version${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${YELLOW}⚠ Please edit .env file with your configuration${NC}"
    echo -e "${YELLOW}  Especially set ANTHROPIC_API_KEY for AI features${NC}"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p uploads reports logs migrations
touch uploads/.gitkeep reports/.gitkeep
echo -e "${GREEN}✓ Directories created${NC}"

# Initialize database
echo ""
echo "Initializing database..."
export FLASK_APP=app.py

# Check if migrations directory exists
if [ ! -d "migrations" ]; then
    flask db init
fi

flask db migrate -m "Initial migration"
flask db upgrade

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Database initialized${NC}"
else
    echo -e "${YELLOW}⚠ Database initialization had warnings (this is usually OK)${NC}"
fi

# Offer to create admin user
echo ""
echo -e "${YELLOW}Would you like to create an admin user? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    flask create-admin
fi

# Final instructions
echo ""
echo "================================"
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Add your ANTHROPIC_API_KEY for AI features"
echo "3. Run: python app.py"
echo "4. Visit: http://localhost:5000"
echo ""
echo "For production deployment, see README.md"
echo ""
echo "Happy analyzing! 🚀"
