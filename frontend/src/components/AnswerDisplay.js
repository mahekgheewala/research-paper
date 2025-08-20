// Answer display component with loading states and formatted output
import React from 'react';
import { Card, Spinner, Alert, Badge } from 'react-bootstrap';

function AnswerDisplay({ answer, loading, question }) {

  // Function to copy answer to clipboard
  const copyToClipboard = () => {
    navigator.clipboard.writeText(answer).then(() => {
      // You could add a toast notification here
      console.log('Answer copied to clipboard');
    }).catch(err => {
      console.error('Failed to copy: ', err);
    });
  };

  return (
    <Card className="answer-display-card shadow-sm mt-4">
      <Card.Header className="d-flex justify-content-between align-items-center">
        <div className="d-flex align-items-center">
          <i className="bi bi-robot me-2" style={{ fontSize: '1.2rem' }}></i>
          <span className="fw-semibold">AI Response</span>
        </div>
        
        {/* Status badge */}
        {loading ? (
          <Badge bg="warning" className="status-badge">
            <Spinner
              as="span"
              animation="border"
              size="sm"
              role="status"
              aria-hidden="true"
              className="me-1"
            />
            Processing
          </Badge>
        ) : (
          <Badge bg="success" className="status-badge">
            <i className="bi bi-check-circle me-1"></i>
            Complete
          </Badge>
        )}
      </Card.Header>

      <Card.Body>
        {/* Show user's question for context */}
        {question && (
          <Alert variant="light" className="question-context mb-3">
            <div className="d-flex">
              <i className="bi bi-person-circle me-2 mt-1 flex-shrink-0"></i>
              <div>
                <strong>Your question:</strong>
                <div className="mt-1 text-muted">{question}</div>
              </div>
            </div>
          </Alert>
        )}

        {/* Loading state */}
        {loading && (
          <div className="text-center py-5 loading-container">
            <Spinner animation="border" variant="primary" className="mb-3" />
            <div className="loading-text">
              <h5>AI is analyzing your question...</h5>
              <p className="text-muted mb-0">This may take a few seconds</p>
            </div>
            
            {/* Loading tips */}
            <div className="mt-4">
              <small className="text-muted">
                <i className="bi bi-lightbulb me-1"></i>
                The AI is searching through research databases and formulating a comprehensive response
              </small>
            </div>
          </div>
        )}

        {/* Answer content */}
        {answer && !loading && (
          <div className="answer-content">
            <div className="d-flex justify-content-between align-items-start mb-3">
              <div className="d-flex align-items-center">
                <i className="bi bi-chat-dots me-2" style={{ fontSize: '1.1rem' }}></i>
                <span className="fw-semibold text-success">Answer</span>
              </div>
              
              {/* Copy button */}
              <button
                className="btn btn-outline-secondary btn-sm copy-btn"
                onClick={copyToClipboard}
                title="Copy to clipboard"
              >
                <i className="bi bi-clipboard"></i>
              </button>
            </div>

            {/* Formatted answer text */}
            <div className="answer-text">
              {answer.split('\n').map((paragraph, index) => (
                paragraph.trim() !== '' && (
                  <p key={index} className="mb-3 lh-base">
                    {paragraph}
                  </p>
                )
              ))}
            </div>

            {/* Answer footer with metadata */}
            <div className="answer-footer mt-4 pt-3 border-top">
              <div className="row align-items-center">
                <div className="col-md-8">
                  <small className="text-muted">
                    <i className="bi bi-clock me-1"></i>
                    Response generated on {new Date().toLocaleString()}
                  </small>
                </div>
                <div className="col-md-4 text-md-end mt-2 mt-md-0">
                  <small className="text-muted">
                    <i className="bi bi-info-circle me-1"></i>
                    Length: {answer.length} characters
                  </small>
                </div>
              </div>
            </div>
          </div>
        )}
      </Card.Body>
    </Card>
  );
}

export default AnswerDisplay;