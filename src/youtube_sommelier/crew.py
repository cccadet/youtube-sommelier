import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from src.youtube_sommelier.tools.youtube import SumarizeVideo, SearchTranscript


model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
llm = ChatOpenAI(temperature=0, model_name=model_name)


@CrewBase
class YoutubeSommelierCrew():
	"""YoutubeSommelier crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def video_sumarizer(self) -> Agent:
		return Agent(
			config=self.agents_config['video_sumarizer'],
			tools=[SumarizeVideo()],
			verbose=True,
			llm=llm,
			allow_delegation=False,
		)

	@agent
	def video_searcher(self) -> Agent:
		return Agent(
			config=self.agents_config['video_searcher'],
    		tools=[SearchTranscript(result_as_answer=True)],
			verbose=True,
			llm=llm,
			allow_delegation=False,
		)

	@task
	def sumarize_video(self) -> Task:
		return Task(
			config=self.tasks_config['sumarize_video'],
			agent=self.video_sumarizer(),
			human_input=True,
		)

	@task
	def search_video(self) -> Task:
		return Task(
			config=self.tasks_config['search_video'],
			agent=self.video_searcher(),
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the YoutubeSommelier crew"""
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=2,
			memory=False,
			llm=llm,
			full_output=True,
			planning=True,
		)