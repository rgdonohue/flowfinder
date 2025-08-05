# FLOWFINDER Benchmark - Code Review Brief

## Project Context
Scientific watershed delineation benchmarking tool. Production-ready but not enterprise SaaS.
Target users: researchers, consultants, government agencies.

## Review Focus
- Code quality and maintainability
- Error handling and edge cases
- Performance bottlenecks with real geospatial data
- Testing gaps that matter
- Documentation clarity

## Explicitly NOT Needed
- Microservices architecture
- Advanced authentication/authorization
- Distributed computing frameworks
- Complex CI/CD pipelines
- Enterprise logging/monitoring systems
- Docker/Kubernetes deployment

## KISS Principle
Prefer simple, readable solutions over clever/complex ones.
This is a scientific tool, not a web application.
