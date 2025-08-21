import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Alert, Spinner } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap-icons/font/bootstrap-icons.css';

// Components
import CustomNavbar from './components/Navbar';
import PaperSearch from './components/PaperSearch';
import PaperSelection from './components/PaperSelection';
import QuestionForm from './components/QuestionForm';
import AnswerDisplay from './components/AnswerDisplay';
// API Configuration
const API_BASE_URL = 'https://research-paper-2.onrender.com/api';


function App() {
  // Main application state
  const [currentStep, setCurrentStep] = useState('search'); // 'search', 'select', 'qa'
  const [searchQuery, setSearchQuery] = useState('');
  const [papers, setPapers] = useState([]);
  const [selectedPapers, setSelectedPapers] = useState([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showAbout, setShowAbout] = useState(false);
  const [sessionStatus, setSessionStatus] = useState(null);

  // API helper function with proper error handling
  const apiCall = async (endpoint, options = {}) => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'API request failed');
      }
      
      return data;
    } catch (error) {
      console.error(`API call failed for ${endpoint}:`, error);
      throw error;
    }
  };

  // Check session status on app load
  useEffect(() => {
    const checkSessionStatus = async () => {
      try {
        const data = await apiCall('/get_session_status');
        // Store session status and update the current step based on it
        setSessionStatus(data);
        
        // If we have papers and they're ready for QA, set appropriate state
        if (data.qa_ready) {
          setCurrentStep('qa');
        } else if (data.papers_count > 0) {
          setCurrentStep('select');
        }
      } catch (error) {
        console.error('Failed to check session status:', error);
        setError('Failed to connect to the server. Please try again.');
      }
    };

    checkSessionStatus();
  }, []);

  // Search for papers
  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;

    setLoading(true);
    setError('');

    try {
      const data = await apiCall('/search_papers', {
        method: 'POST',
        body: JSON.stringify({
          query: searchQuery,
          max_results: 8
        }),
      });

      setPapers(data.papers || []);
      setCurrentStep('select');
    } catch (error) {
      setError(error.message || 'Failed to search papers');
    } finally {
      setLoading(false);
    }
  };

  // Select papers for processing
  const handlePaperSelection = async (selectedIndices) => {
    setLoading(true);
    setError('');

    try {
      const data = await apiCall('/select_papers', {
        method: 'POST',
        body: JSON.stringify({
          selected_papers: selectedIndices
        }),
      });

      setSelectedPapers(data.papers_info || []);
      setCurrentStep('qa');
    } catch (error) {
      setError(error.message || 'Failed to process selected papers');
    } finally {
      setLoading(false);
    }
  };

  // Ask question
  const handleQuestionSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError('');
    setAnswer('');

    try {
      const data = await apiCall('/ask_question', {
        method: 'POST',
        body: JSON.stringify({
          question: question
        }),
      });

      setAnswer(data.answer || 'No answer received');
    } catch (error) {
      setError(error.message || 'Failed to get answer');
    } finally {
      setLoading(false);
    }
  };

  // Go back to paper selection
  const handleBackToSelection = () => {
    setCurrentStep('select');
    setQuestion('');
    setAnswer('');
    setError('');
  };

  // Go back to search
  const handleBackToSearch = () => {
    setCurrentStep('search');
    setPapers([]);
    setSelectedPapers([]);
    setQuestion('');
    setAnswer('');
    setError('');
  };

  // Clear all and start over
  const handleClearAll = async () => {
    setLoading(true);
    try {
      await apiCall('/clear_session', { method: 'POST' });
      
      // Reset all state
      setCurrentStep('search');
      setSearchQuery('');
      setPapers([]);
      setSelectedPapers([]);
      setQuestion('');
      setAnswer('');
      setError('');
      setSessionStatus(null);
    } catch (error) {
      setError('Failed to clear session');
    } finally {
      setLoading(false);
    }
  };

  // Toggle about section
  const toggleAbout = () => {
    setShowAbout(!showAbout);
  };

  return (
    <div className="App">
      {/* Show connection status if there's an issue */}
      {sessionStatus && !sessionStatus.success && (
        <Alert variant="warning" className="m-2">
          Connection to server lost. Please refresh the page.
        </Alert>
      )}
      <CustomNavbar 
        onClear={handleClearAll} 
        onToggleAbout={toggleAbout}
        showAbout={showAbout}
      />
      
      <Container className="mt-4">
        {/* Error display */}
        {error && (
          <Alert variant="danger" dismissible onClose={() => setError('')}>
            <Alert.Heading>Error</Alert.Heading>
            {error}
          </Alert>
        )}

        {/* About section */}
        {showAbout && (
          <Row className="mb-4">
            <Col>
              <Alert variant="info">
                <Alert.Heading>About AI Research Q&A</Alert.Heading>
                <p>
                  This application allows you to search for research papers from arXiv, 
                  select the ones you're interested in, and then ask detailed questions 
                  about their content. The AI will analyze the papers and provide 
                  comprehensive answers based on the research.
                </p>
                <hr />
                <h6>How to use:</h6>
                <ol>
                  <li><strong>Search:</strong> Enter keywords to find relevant papers from arXiv</li>
                  <li><strong>Select:</strong> Choose the papers you want to analyze</li>
                  <li><strong>Ask:</strong> Submit questions about the selected papers</li>
                </ol>
                <p className="mb-0">
                  <small>
                    <strong>Note:</strong> Processing papers may take a few minutes depending on 
                    their size and complexity.
                  </small>
                </p>
              </Alert>
            </Col>
          </Row>
        )}

        {/* Main content based on current step */}
        <Row>
          <Col>
            {/* Step 1: Paper Search */}
            {currentStep === 'search' && (
              <PaperSearch
                searchQuery={searchQuery}
                setSearchQuery={setSearchQuery}
                onSubmit={handleSearchSubmit}
                loading={loading}
              />
            )}

            {/* Step 2: Paper Selection */}
            {currentStep === 'select' && (
              <PaperSelection
                papers={papers}
                onSelectPapers={handlePaperSelection}
                onBack={handleBackToSearch}
                loading={loading}
                searchQuery={searchQuery}
              />
            )}

            {/* Step 3: Q&A Interface */}
            {currentStep === 'qa' && (
              <>
                {/* Selected papers summary */}
                {selectedPapers.length > 0 && (
                  <Alert variant="success" className="mb-4">
                    <Alert.Heading as="h6">
                      <i className="bi bi-check-circle me-2"></i>
                      Papers Ready for Analysis
                    </Alert.Heading>
                    <p className="mb-2">
                      Successfully processed <strong>{selectedPapers.length}</strong> papers:
                    </p>
                    <ul className="mb-2">
                      {selectedPapers.slice(0, 3).map((paper, index) => (
                        <li key={index}>
                          <small>
                            {paper.title} - {paper.authors}
                          </small>
                        </li>
                      ))}
                      {selectedPapers.length > 3 && (
                        <li><small>...and {selectedPapers.length - 3} more papers</small></li>
                      )}
                    </ul>
                    <div className="d-flex gap-2">
                      <button 
                        className="btn btn-outline-secondary btn-sm"
                        onClick={handleBackToSelection}
                        disabled={loading}
                      >
                        <i className="bi bi-arrow-left me-1"></i>
                        Change Selection
                      </button>
                      <button 
                        className="btn btn-outline-primary btn-sm"
                        onClick={handleBackToSearch}
                        disabled={loading}
                      >
                        <i className="bi bi-search me-1"></i>
                        New Search
                      </button>
                    </div>
                  </Alert>
                )}

                {/* Question form */}
                <QuestionForm
                  question={question}
                  setQuestion={setQuestion}
                  onSubmit={handleQuestionSubmit}
                  loading={loading}
                />

                {/* Answer display */}
                {(answer || loading) && (
                  <AnswerDisplay
                    answer={answer}
                    loading={loading}
                    question={question}
                  />
                )}
              </>
            )}

            {/* Loading overlay for major operations */}
            {loading && currentStep !== 'qa' && (
              <div className="text-center py-5">
                <Spinner animation="border" variant="primary" size="lg" />
                <div className="mt-3">
                  <h5>Processing...</h5>
                  <p className="text-muted">
                    {currentStep === 'search' ? 'Searching arXiv database...' : 
                     currentStep === 'select' ? 'Processing selected papers...' : 
                     'Please wait...'}
                  </p>
                </div>
              </div>
            )}
          </Col>
        </Row>

        {/* Footer */}
        <Row className="mt-5">
          <Col>
            <div className="text-center text-muted py-3 border-top">
              <small>
                <i className="bi bi-robot me-1"></i>
                AI Research Q&A System - Powered by arXiv and Gemini AI
              </small>
            </div>
          </Col>
        </Row>
      </Container>

      {/* Custom styles */}
      <style jsx>{`
        .custom-navbar {
          border-bottom: 1px solid #dee2e6;
        }
        
        .search-card {
          border-radius: 10px;
          border: 1px solid #e9ecef;
        }
        
        .search-title {
          color: #495057;
        }
        
        .search-btn {
          border-radius: 8px;
          font-weight: 500;
        }
        
        .paper-card {
          border-radius: 8px;
          border: 1px solid #e9ecef;
          transition: all 0.2s ease;
        }
        
        .paper-card:hover {
          border-color: #007bff;
          box-shadow: 0 2px 8px rgba(0,123,255,0.15);
        }
        
        .paper-card.selected {
          border-color: #28a745;
          background-color: #f8fff9;
        }
        
        .question-form-card {
          border-radius: 10px;
          border: 1px solid #e9ecef;
        }
        
        .question-textarea {
          border-radius: 8px;
        }
        
        .answer-display-card {
          border-radius: 10px;
          border: 1px solid #e9ecef;
        }
        
        .answer-text {
          line-height: 1.7;
          color: #495057;
        }
        
        .tips-list {
          font-size: 0.9rem;
          color: #6c757d;
        }
        
        .example-btn {
          font-size: 0.8rem;
          border-radius: 15px;
        }
        
        .copy-btn {
          border-radius: 6px;
        }
        
        .status-badge {
          font-size: 0.75rem;
        }
        
        .sticky-bottom {
          position: sticky;
          bottom: 0;
          z-index: 10;
        }
      `}</style>
    </div>
  );
}

export default App;