// Custom navigation bar component with branding and clear functionality
import React from 'react';
import { Navbar, Nav, Container, Button } from 'react-bootstrap';

function CustomNavbar({ onClear, onToggleAbout, showAbout }) {
  return (
    <Navbar bg="dark" variant="dark" expand="lg" className="custom-navbar">
      <Container>
        {/* Brand/Logo section */}
        <Navbar.Brand href="#" className="d-flex align-items-center">
          <i className="bi bi-robot me-2" style={{ fontSize: '1.5rem' }}></i>
          <span className="fw-bold">AI Research Q&A</span>
        </Navbar.Brand>

        {/* Collapsible navigation for mobile */}
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          
          {/* Navigation links */}
          <Nav className="me-auto">
            <Nav.Link 
              href="#" 
              className="nav-link-custom"
              onClick={(e) => {
                e.preventDefault();
                if (showAbout) onToggleAbout();
              }}
            >
              Home
            </Nav.Link>
            <Nav.Link 
              href="#" 
              className={`nav-link-custom ${showAbout ? 'active' : ''}`}
              onClick={(e) => {
                e.preventDefault();
                onToggleAbout();
              }}
            >
              About
            </Nav.Link>
          </Nav>

          {/* Right-aligned clear button */}
          <Nav>
            <Button 
              variant="outline-light" 
              size="sm" 
              onClick={onClear}
              className="clear-btn"
            >
              <i className="bi bi-arrow-clockwise me-1"></i>
              New Question
            </Button>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}

export default CustomNavbar;