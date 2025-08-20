// Question input form component with validation and submit functionality
import React from 'react';
import { Form, Button, InputGroup, Card } from 'react-bootstrap';

function QuestionForm({ question, setQuestion, onSubmit, loading }) {
  
  // Handle input changes
  const handleInputChange = (e) => {
    setQuestion(e.target.value);
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(e);
  };

  // Handle Enter key press (submit form)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit(e);
    }
  };

  return (
    <Card className="question-form-card shadow-sm">
      <Card.Body>
        <Form onSubmit={handleSubmit}>
          
          {/* Question input section */}
          <Form.Group className="mb-3">
            <Form.Label className="fw-semibold text-dark">
              <i className="bi bi-question-circle me-2"></i>
              Ask your research question
            </Form.Label>
            
            {/* Input group with icon and textarea */}
            <InputGroup className="question-input-group">
              <Form.Control
                as="textarea"
                rows={3}
                placeholder="Enter your research question here... (e.g., 'What are the latest developments in renewable energy?')"
                value={question}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                disabled={loading}
                className="question-textarea"
                style={{ 
                  resize: 'vertical',
                  minHeight: '80px'
                }}
              />
            </InputGroup>
            
            {/* Character count helper */}
            <Form.Text className="text-muted">
              {question.length}/1000 characters
            </Form.Text>
          </Form.Group>

          {/* Submit button section */}
          <div className="d-grid gap-2 d-md-flex justify-content-md-end">
            <Button
              variant="primary"
              type="submit"
              disabled={loading || !question.trim()}
              size="lg"
              className="submit-btn px-4"
            >
              {loading ? (
                <>
                  {/* Loading spinner */}
                  <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                  Thinking...
                </>
              ) : (
                <>
                  <i className="bi bi-send me-2"></i>
                  Ask AI
                </>
              )}
            </Button>
          </div>

          {/* Help text */}
          <div className="mt-3">
            <small className="text-muted">
              <i className="bi bi-info-circle me-1"></i>
              Tip: Be specific and clear in your question for better results. You can use Shift+Enter for new lines.
            </small>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );
}

export default QuestionForm;