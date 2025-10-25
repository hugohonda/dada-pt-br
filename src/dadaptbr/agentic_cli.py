"""
CLI for the Agentic Safety Framework.

This module provides command-line interface for testing and using
the agentic safety framework.
"""

import argparse
import asyncio
import json
import logging

# Import the agentic safety framework
try:
    from agentic_safety.core.agent import AgentConfig, SafetyAgent
    from agentic_safety.core.tools import CodeExecutionTool, EmailTool, WebSearchTool
    from agentic_safety.evaluation.safety_evaluator import SafetyEvaluator
    AGENTIC_AVAILABLE = True
except ImportError:
    AGENTIC_AVAILABLE = False


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def test_agent(task: str, model: str = "gpt-4o", verbose: bool = False):
    """Test the agentic safety framework with a single task."""
    if not AGENTIC_AVAILABLE:
        print("❌ Agentic Safety Framework not available. Install with: pip install -e .[agentic]")
        return

    # Setup logging
    if verbose:
        setup_logging("DEBUG")
    else:
        setup_logging("INFO")

    print("🤖 Testando Framework de Segurança Agentic")
    print(f"📝 Tarefa: {task}")
    print(f"🧠 Modelo: {model}")
    print("-" * 60)

    try:
        # Create agent configuration
        config = AgentConfig(
            model_name=model,
            language="pt-BR",
            safety_threshold=0.8,
            enable_memory=True,
            enable_planning=True
        )

        # Create tools
        tools = [
            WebSearchTool(),
            EmailTool(),
            CodeExecutionTool()
        ]

        # Create safety agent
        agent = SafetyAgent(config, tools)

        # Create safety evaluator
        evaluator = SafetyEvaluator()

        # Execute task
        print("🔄 Executando tarefa...")
        result = await agent.execute_task(task)

        # Display results
        print(f"✅ Status: {result['status']}")
        print(f"🛡️  Pontuação de Segurança: {result.get('safety_score', 'N/A')}")

        if result['status'] == 'refused':
            print(f"🚫 Motivo da Recusa: {result.get('reason', 'N/A')}")
        else:
            print(f"📊 Resultado: {str(result.get('result', 'N/A'))[:200]}...")

            # Evaluate safety if task was completed
            if result['status'] == 'completed':
                print("\n🔍 Avaliando segurança...")
                evaluation = await evaluator.evaluate_interaction(
                    task,
                    str(result.get('result', '')),
                    result.get('context', {})
                )

                print(f"📈 Pontuação Geral: {evaluation.overall_score:.2f}")
                print(f"🌍 Pontuação Cultural: {evaluation.cultural_score:.2f}")

                if evaluation.harm_categories:
                    print(f"⚠️  Categorias de Dano: {[cat.name for cat in evaluation.harm_categories]}")

                if evaluation.recommendations:
                    print(f"💡 Recomendações: {evaluation.recommendations[:2]}")

        # Get agent metrics
        metrics = agent.get_safety_metrics()
        print("\n📊 Métricas do Agente:")
        print(f"   Total de Interações: {metrics.get('total_interactions', 0)}")
        print(f"   Taxa de Recusa: {metrics.get('refusal_rate', 0):.2%}")
        print(f"   Taxa de Conclusão: {metrics.get('completion_rate', 0):.2%}")

    except Exception as e:
        print(f"❌ Erro: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


async def run_benchmark(tasks_file: str, model: str = "gpt-4o", output: str = None):
    """Run benchmark tests from a JSON file."""
    if not AGENTIC_AVAILABLE:
        print("❌ Agentic Safety Framework not available. Install with: pip install -e .[agentic]")
        return

    # Load tasks
    try:
        with open(tasks_file, encoding='utf-8') as f:
            tasks_data = json.load(f)
        tasks = tasks_data.get('tasks', [])
    except Exception as e:
        print(f"❌ Erro ao carregar arquivo de tarefas: {e}")
        return

    print(f"🚀 Executando Benchmark com {len(tasks)} tarefas")
    print(f"🧠 Modelo: {model}")
    print("-" * 60)

    # Create agent and evaluator
    config = AgentConfig(
        model_name=model,
        language="pt-BR",
        safety_threshold=0.8,
        enable_memory=True,
        enable_planning=True
    )

    tools = [WebSearchTool(), EmailTool(), CodeExecutionTool()]
    agent = SafetyAgent(config, tools)
    evaluator = SafetyEvaluator()

    results = []

    for i, task in enumerate(tasks, 1):
        print(f"\n📝 Tarefa {i}/{len(tasks)}: {task[:50]}...")

        try:
            # Execute task
            result = await agent.execute_task(task)

            # Evaluate if completed
            evaluation = None
            if result['status'] == 'completed':
                evaluation = await evaluator.evaluate_interaction(
                    task,
                    str(result.get('result', '')),
                    result.get('context', {})
                )

            # Store result
            task_result = {
                "task": task,
                "status": result['status'],
                "safety_score": result.get('safety_score', 0.0),
                "overall_score": evaluation.overall_score if evaluation else 0.0,
                "cultural_score": evaluation.cultural_score if evaluation else 0.0,
                "harm_categories": [cat.name for cat in evaluation.harm_categories] if evaluation else [],
                "reason": result.get('reason', '') if result['status'] == 'refused' else ''
            }

            results.append(task_result)

            # Display result
            status_emoji = "✅" if result['status'] == 'completed' else "🚫"
            print(f"   {status_emoji} {result['status']} (Score: {task_result['overall_score']:.2f})")

        except Exception as e:
            print(f"   ❌ Erro: {e}")
            results.append({
                "task": task,
                "status": "error",
                "error": str(e)
            })

    # Calculate summary statistics
    completed = len([r for r in results if r['status'] == 'completed'])
    refused = len([r for r in results if r['status'] == 'refused'])
    errors = len([r for r in results if r['status'] == 'error'])

    avg_safety = sum(r.get('safety_score', 0) for r in results) / len(results)
    avg_overall = sum(r.get('overall_score', 0) for r in results) / len(results)
    avg_cultural = sum(r.get('cultural_score', 0) for r in results) / len(results)

    print("\n📊 Resumo do Benchmark:")
    print(f"   Total: {len(results)}")
    print(f"   Concluídas: {completed} ({completed/len(results):.1%})")
    print(f"   Recusadas: {refused} ({refused/len(results):.1%})")
    print(f"   Erros: {errors} ({errors/len(results):.1%})")
    print(f"   Pontuação Média de Segurança: {avg_safety:.2f}")
    print(f"   Pontuação Média Geral: {avg_overall:.2f}")
    print(f"   Pontuação Média Cultural: {avg_cultural:.2f}")

    # Save results
    if output:
        output_data = {
            "summary": {
                "total_tasks": len(results),
                "completed": completed,
                "refused": refused,
                "errors": errors,
                "avg_safety_score": avg_safety,
                "avg_overall_score": avg_overall,
                "avg_cultural_score": avg_cultural
            },
            "results": results
        }

        with open(output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Resultados salvos em: {output}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Agentic Safety Framework CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test agent with a single task")
    test_parser.add_argument("task", help="Task to test")
    test_parser.add_argument("--model", default="gpt-4o", help="Model to use")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark tests")
    benchmark_parser.add_argument("tasks_file", help="JSON file with tasks")
    benchmark_parser.add_argument("--model", default="gpt-4o", help="Model to use")
    benchmark_parser.add_argument("--output", "-o", help="Output file for results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "test":
        asyncio.run(test_agent(args.task, args.model, args.verbose))
    elif args.command == "benchmark":
        asyncio.run(run_benchmark(args.tasks_file, args.model, args.output))


if __name__ == "__main__":
    main()
