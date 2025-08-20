import React, { useState } from 'react';
import { Card, Button, Form, Badge, Row, Col, Alert } from 'react-bootstrap';

const PaperSelection = ({ papers, onSelectPapers, onBack, loading, searchQuery }) => {
  const [selectedPapers, setSelectedPapers] = useState(new Set());
  const [selectAll, setSelectAll] = useState(false);

  // Handle individual paper selection
  const handlePaperToggle = (paperId) => {
    const newSelected = new Set(selectedPapers);
    if (newSelected.has(paperId)) {
      newSelected.delete(paperId);
    } else {
      newSelected.add(paperId);
    }
    setSelectedPapers(newSelected);
    setSelectAll(newSelected.size === papers.length);
  };

  // Handle select all toggle
  const handleSelectAll = () => {
    if (selectAll) {
      setSelectedPapers(new Set());
      setSelectAll(false);
    } else {
      setSelectedPapers(new Set(papers.map(paper => paper.id)));
      setSelectAll(true);
    }
  };

  // Handle paper selection submission
  const handleSubmit = () => {
    if (selectedPapers.size === 0) {
      return;
    }
    onSelectPapers(Array.from(selectedPapers));
  };

  // Truncate text helper
  const truncateText = (text, maxLength = 300) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  return (
    <div className="paper-selection">
      {/* Header */}
      <div className="selection-header mb-4">
        <div className="d-flex justify-content-between align-items-center mb-3">
          <div>
            <h3>
              <i className="bi bi-file-text me-2"></i>
              Select Papers to Analyze
            </h3>
            <p className="text-muted mb-0">
              Found <strong>{papers.length}</strong> papers for "{searchQuery}"
            </p>
          </div>
          <Button variant="outline-secondary" onClick={onBack} disabled={loading}>
            <i className="bi bi-arrow-left me-2"></i>
            New Search
          </Button>
        </div>

        {/* Selection controls */}
        <div className="selection-controls mb-3">
          <Row className="align-items-center">
            <Col md={6}>
              <Form.Check
                type="checkbox"
                label={`Select All Papers (${papers.length})`}
                checked={selectAll}
                onChange={handleSelectAll}
                disabled={loading}
                className="select-all-checkbox"
              />
            </Col>
            <Col md={6} className="text-md-end">
              <Badge bg="primary" className="me-2">
                {selectedPapers.size} Selected
              </Badge>
              <Button
                variant="success"
                disabled={selectedPapers.size === 0 || loading}
                onClick={handleSubmit}
                className="process-btn"
              >
                {loading ? (
                  <>
                    <span 
                      className="spinner-border spinner-border-sm me-2" 
                      role="status" 
                      aria-hidden="true"
                    ></span>
                    Processing Papers...
                  </>
                ) : (
                  <>
                    <i className="bi bi-gear me-2"></i>
                    Process Selected Papers ({selectedPapers.size})
                  </>
                )}
              </Button>
            </Col>
          </Row>
        </div>

        {/* Info alert */}
        <Alert variant="info" className="mb-4">
          <Alert.Heading as="h6">
            <i className="bi bi-info-circle me-2"></i>
            How it works:
          </Alert.Heading>
          Select the papers you want to analyze. The system will download and process the full content, 
          then you can ask detailed questions about them. Processing may take a few minutes depending 
          on the number and size of papers.
        </Alert>
      </div>

      {/* Papers list */}
      <div className="papers-list">
        {papers.map((paper) => (
          <Card 
            key={paper.id} 
            className={`paper-card mb-3 ${selectedPapers.has(paper.id) ? 'selected' : ''}`}
            style={{ cursor: 'pointer' }}
          >
            <Card.Body>
              <Row>
                <Col xs={1} className="d-flex align-items-start pt-1">
                  <Form.Check
                    type="checkbox"
                    checked={selectedPapers.has(paper.id)}
                    onChange={() => handlePaperToggle(paper.id)}
                    disabled={loading}
                    className="paper-checkbox"
                  />
                </Col>
                <Col xs={11} onClick={() => !loading && handlePaperToggle(paper.id)}>
                  <div className="paper-header mb-2">
                    <h5 className="paper-title mb-2">{paper.title}</h5>
                    <div className="paper-meta mb-2">
                      <Badge bg="secondary" className="me-2">
                        <i className="bi bi-file-pdf me-1"></i>
                        {paper.arxiv_id}
                      </Badge>
                      {paper.published && (
                        <Badge bg="outline-secondary" className="me-2">
                          <i className="bi bi-calendar me-1"></i>
                          {paper.published}
                        </Badge>
                      )}
                      <Badge bg="outline-info">
                        <i className="bi bi-people me-1"></i>
                        {paper.authors}
                      </Badge>
                    </div>
                  </div>

                  {paper.summary && (
                    <div className="paper-summary mb-3">
                      <p className="summary-text mb-0">
                        {truncateText(paper.summary)}
                      </p>
                    </div>
                  )}

                  <div className="paper-actions">
                    <Row className="align-items-center">
                      <Col>
                        <div className="paper-links">
                          {paper.arxiv_url && (
                            <a 
                              href={paper.arxiv_url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="btn btn-outline-primary btn-sm me-2"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <i className="bi bi-box-arrow-up-right me-1"></i>
                              View on arXiv
                            </a>
                          )}
                          {paper.pdf_url && (
                            <a 
                              href={paper.pdf_url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="btn btn-outline-secondary btn-sm"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <i className="bi bi-file-pdf me-1"></i>
                              Download PDF
                            </a>
                          )}
                        </div>
                      </Col>
                      <Col xs="auto">
                        {selectedPapers.has(paper.id) && (
                          <Badge bg="success">
                            <i className="bi bi-check-circle me-1"></i>
                            Selected
                          </Badge>
                        )}
                      </Col>
                    </Row>
                  </div>
                </Col>
              </Row>
            </Card.Body>
          </Card>
        ))}
      </div>

      {/* Bottom action bar */}
      {papers.length > 3 && (
        <div className="bottom-action-bar sticky-bottom bg-white p-3 border-top">
          <Row className="align-items-center">
            <Col>
              <span className="selected-count">
                <i className="bi bi-check-square me-2"></i>
                {selectedPapers.size} of {papers.length} papers selected
              </span>
            </Col>
            <Col xs="auto">
              <Button
                variant="success"
                disabled={selectedPapers.size === 0 || loading}
                onClick={handleSubmit}
                size="lg"
              >
                {loading ? (
                  <>
                    <span 
                      className="spinner-border spinner-border-sm me-2" 
                      role="status" 
                      aria-hidden="true"
                    ></span>
                    Processing...
                  </>
                ) : (
                  <>
                    <i className="bi bi-gear me-2"></i>
                    Process {selectedPapers.size} Papers
                  </>
                )}
              </Button>
            </Col>
          </Row>
        </div>
      )}

      {/* No papers message */}
      {papers.length === 0 && (
        <div className="text-center py-5">
          <i className="bi bi-search display-1 text-muted"></i>
          <h4 className="mt-3">No papers found</h4>
          <p className="text-muted">Try different keywords or broader search terms</p>
          <Button variant="primary" onClick={onBack}>
            <i className="bi bi-arrow-left me-2"></i>
            Try New Search
          </Button>
        </div>
      )}
    </div>
  );
};

export default PaperSelection;