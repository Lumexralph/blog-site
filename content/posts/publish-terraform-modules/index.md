+++
draft = true
date = 2023-10-08
title = "Publish Terraform modules to a Terraform private registry from monorepo"
description = "automate the semantic release of terraform modules from a monorepo to terraform cloud private registry"
slug = ""
authors = ["Olumide Ogundele"]
tags = ["terraform", "infra", "devops"]
categories = ["devops", "terraform"]
externalLink = ""
series = ["cloud infrastructure"]
+++

I decided to make a note about this looking at how there was no easy way of doing this just because sometimes not
every tool provider thinks about using their tool in a monorepo.

I have experienced both worlds of distributed repositories and monorepo (having all your projects in a single repository,
one codebase for everything), companies like Google, Meta, Uber and many more use a monorepo. They both have their quirks,
but I am currently enjoying working with a monorepo using [Bazel](https://bazel.build/) as the build tool.


## What is Terraform ?

If you've ever heard the word Infrastructure as Code ([IaC](https://learn.microsoft.com/en-us/devops/deliver/what-is-infrastructure-as-code)),
[Terraform](https://developer.hashicorp.com/terraform/docs) is a popular tool used for automation of infrastructure
to provision and manage resources in any cloud or data center using a declarative configuration language (HashiCorp Configuration Language - HCL)
which can be versioned-controlled and treated like any other codebase, which is part of [GitOps](https://about.gitlab.com/topics/gitops/).

## What is a Terraform module ?

According to [terraform](https://developer.hashicorp.com/terraform/tutorials/modules/pattern-module-creation)
> Terraform modules are self-contained pieces of infrastructure-as-code that abstract the underlying complexity of infrastructure deployments

We use modules to abstract common IaC and follow a DRY pattern with our IaC. Our infrastructure state is deployed on
[Terraform Cloud](https://www.hashicorp.com/products/terraform) and the state is split into smaller states by
using [workspaces](https://developer.hashicorp.com/terraform/cloud-docs/workspaces/creating) broken into something
like `dev`, `staging` and `prod`, you can go a bit granular in your project.

### What is Private Registry ?

Think of this like Docker registry, NPM registry and more, just that it's for terraform resources and it's private, not
opened to public access.

### Why publish your Terraform modules ?

The modules are shared across workspaces and used by different people in the team. A situation will arise that will
warrant updating the module to meet new requirements and this will lead to a breaking change across workspaces using
the module, hence, you'll have to figure out a way to version the modules and this led to the reason for this writeup.
A way to publish and semantically version the modules in a monorepo.

## The challenge of Terraform modules in a monorepo ?

The issue that's yet to be solved by terraform cloud is that they assume you'll have your modules in separate repositories.
I don't know how feasible this will be, imagine you have like 20 terraform modules, does that mean you'll have to create
20 separate repositories? It just doesn't work for our case anyway so this requires some extra thoughts and logic, and
it was important for me to have a very simple solution, I don't want to spin up crazy infra for this.

## How I automated semantic versioning and release of our Terraform modules ?

To achieve this I used GitHub Actions, SemVar and Terraform Cloud private registry API with Python.

To create a new module, it's important to follow the [standard structure](https://developer.hashicorp.com/terraform/language/modules/develop/structure)

```md
├── main.tf
├── outputs.tf
├── README.md
└── variables.tf
```